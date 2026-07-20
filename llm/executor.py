"""Unified LLM executor supporting OpenAI and Brainpowa backends."""

from __future__ import annotations

import asyncio
import os
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv

_dltk_config = Path.home() / ".dltk.config"
if _dltk_config.exists():
    load_dotenv(_dltk_config)
load_dotenv()

from typing import TYPE_CHECKING, Any, Sequence, TypeVar, overload

from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel

from utils.vis_image import ImageContentCache, get_image_content, image_to_base64

if TYPE_CHECKING:
    pass

T = TypeVar("T", bound=BaseModel)


def _get_openai_api_key(passed_key: str | None = None) -> str:
    """Dynamically retrieve OpenAI API key, reloading ~/.dltk.config if needed."""
    if passed_key:
        return passed_key
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        dltk_cfg = Path.home() / ".dltk.config"
        if dltk_cfg.exists():
            load_dotenv(dltk_cfg)
        key = os.getenv("OPENAI_API_KEY", "")
    return key


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Reasoning models do not support temperature, top_p, or similar sampling params.
# Sending them can cause API errors or silent degradation.
_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def _is_reasoning_model(model_name: str) -> bool:
    return any(model_name.startswith(prefix) for prefix in _REASONING_MODEL_PREFIXES)


class LLMExecutor:
    """Unified executor wrapping LLM clients (Brainpowa or ChatOpenAI).

    Example:
        >>> class MathAnswer(BaseModel):
        ...     answer: int
        ...     explanation: str
        >>> executor = LLMExecutor.from_model_name("gpt-4o-mini")
        >>> result = executor.predict(
        ...     [HumanMessage(content="What is 2+2?")],
        ...     output_object_type=MathAnswer,
        ... )
        >>> result.answer
        4
    """

    def __init__(
        self,
        client: ChatOpenAI,
        image_content_cache: ImageContentCache | None = None,
    ):
        self.client = client
        self.image_content_cache = image_content_cache

    def _build_image_blocks(self, images: Sequence[str | bytes]) -> list[dict[str, Any]]:
        image_blocks = []
        for img in images:
            if isinstance(img, str):
                # URL - fetch and encode
                if self.image_content_cache:
                    image_content = self.image_content_cache.get(img)
                else:
                    image_content = get_image_content(img)
                if image_content is None:
                    logger.warning("Failed to fetch image: {}", img)
                    continue
            else:
                # Raw bytes
                image_content = img

            b64 = image_to_base64(BytesIO(image_content))
            image_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
        return image_blocks

    def _prepare_messages(
        self,
        messages: list[BaseMessage],
        images: Sequence[str | bytes] | None = None,
    ) -> list[BaseMessage]:
        """Prepare messages for LLM invocation, appending images if provided."""
        result = list(messages)

        if not images:
            return result

        image_blocks = self._build_image_blocks(images)
        if image_blocks:
            result.append(HumanMessage(content=image_blocks))

        return result

    def _invoke_client(
        self,
        llm_messages: list[BaseMessage],
        output_object_type: type[T] | None = None,
    ) -> str | T:
        """Invoke the LLM client, returning string or Pydantic model."""
        if output_object_type is not None:
            # Both backends' with_structured_output() return Pydantic model directly
            structured_client = self.client.with_structured_output(output_object_type)
            return structured_client.invoke(llm_messages)
        else:
            # Simple text response - both backends return AIMessage
            response = self.client.invoke(llm_messages)
            return response.content

    @overload
    def predict(
        self,
        messages: list[BaseMessage],
        *,
        images: Sequence[str | bytes] | None = None,
        output_object_type: type[T],
    ) -> T: ...

    @overload
    def predict(
        self,
        messages: list[BaseMessage],
        *,
        images: Sequence[str | bytes] | None = None,
        output_object_type: None = None,
    ) -> str: ...

    def predict(
        self,
        messages: list[BaseMessage],
        *,
        images: Sequence[str | bytes] | None = None,
        output_object_type: type[T] | None = None,
    ) -> str | T:
        """Synchronous LLM prediction."""
        llm_messages = self._prepare_messages(
            messages,
            images=images,
        )
        return self._invoke_client(llm_messages, output_object_type)

    @overload
    async def apredict(
        self,
        messages: list[BaseMessage],
        *,
        images: Sequence[str | bytes] | None = None,
        output_object_type: type[T],
    ) -> T: ...

    @overload
    async def apredict(
        self,
        messages: list[BaseMessage],
        *,
        images: Sequence[str | bytes] | None = None,
        output_object_type: None = None,
    ) -> str: ...

    async def apredict(
        self,
        messages: list[BaseMessage],
        *,
        images: Sequence[str | bytes] | None = None,
        output_object_type: type[T] | None = None,
    ) -> str | T:
        """Asynchronous LLM prediction (thread-wrapped)."""
        return await asyncio.to_thread(
            self.predict,
            messages,
            images=images,
            output_object_type=output_object_type,
        )

    # ==================== Factory Methods ====================

    @staticmethod
    def from_model_name(
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,
        max_retries: int = 10,
        max_tokens: int = 2048,
        reasoning_effort: str | None = None,
        image_content_cache: ImageContentCache | None = None,
        **kwargs,
    ) -> LLMExecutor:
        """Create an executor, auto-detecting backend from model name.

        Args:
            model_name: Model identifier (OpenAI model name).
            temperature: Sampling temperature. Ignored for reasoning models.
            max_retries: Maximum retry attempts.
            max_tokens: Maximum tokens in response.
            reasoning_effort: For reasoning models only ('low', 'medium', 'high').
                Controls how much the model reasons. Defaults to 'high' for reasoning
                models when not specified, as tasks such as image analysis benefit from
                deeper reasoning.
            image_content_cache: Optional cache for image content.
            **kwargs: Additional arguments passed to the backend client.

        Returns:
            LLMExecutor instance configured with the appropriate backend.

        Raises:
            ValueError: If required API key is not set.
        """
        api_key = _get_openai_api_key(kwargs.pop("openai_api_key", None))
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        if _is_reasoning_model(model_name):
            # Reasoning models do not support temperature or top_p — use
            # disabled_params to prevent langchain from sending them at all.
            # reasoning_effort controls how deeply the model reasons;
            # default to 'high' for tasks like image analysis.
            client = ChatOpenAI(
                model_name=model_name,
                openai_api_key=api_key,
                max_retries=max_retries,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort or "high",
                disabled_params={"temperature": None, "top_p": None},
                **kwargs,
            )
        else:
            client = ChatOpenAI(
                model_name=model_name,
                openai_api_key=api_key,
                temperature=temperature,
                max_retries=max_retries,
                top_p=1,
                max_tokens=max_tokens,
                **kwargs,
            )
        return LLMExecutor(client, image_content_cache=image_content_cache)


class ExecutorRegistry:
    """Singleton registry for LLMExecutor instances, cached by configuration.

    This avoids creating duplicate executors with the same configuration
    across multiple modules. Thread-safe via GIL for dict operations.

    Example:
        >>> executor = ExecutorRegistry.get("gpt-4o-mini")
        >>> executor2 = ExecutorRegistry.get("gpt-4o-mini")
        >>> executor is executor2  # Same instance
        True
    """

    _instances: dict[str, LLMExecutor] = {}

    @classmethod
    def get(
        cls,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,
        max_retries: int = 10,
        max_tokens: int = 2048,
        reasoning_effort: str | None = None,
        image_content_cache: ImageContentCache | None = None,
    ) -> LLMExecutor:
        """Get or create an LLMExecutor with the given configuration.

        Executors are cached by (model_name, temperature, max_retries, max_tokens, reasoning_effort).
        Note: image_content_cache is NOT part of the cache key since it's mutable.

        Args:
            model_name: Model identifier (OpenAI model name).
            temperature: Sampling temperature. Ignored for reasoning models.
            max_retries: Maximum retry attempts.
            max_tokens: Maximum tokens in response.
            reasoning_effort: For reasoning models only ('low', 'medium', 'high').
            image_content_cache: Optional cache for image content.

        Returns:
            Cached or newly created LLMExecutor instance.
        """
        key = f"{model_name}:{temperature}:{max_retries}:{max_tokens}:{reasoning_effort}"
        if key not in cls._instances:
            cls._instances[key] = LLMExecutor.from_model_name(
                model_name=model_name,
                temperature=temperature,
                max_retries=max_retries,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                image_content_cache=image_content_cache,
            )
        return cls._instances[key]

    @classmethod
    def clear(cls) -> None:
        """Clear all cached executors. Useful for testing."""
        cls._instances.clear()

import base64
import hashlib
import os
import re
import tempfile
import time
import urllib.request
from io import BytesIO
from typing import Optional

import numpy as np
import requests
from loguru import logger
from path import Path
from PIL import Image

CACHE_FOLDER = os.environ.get("CACHE_FOLDER", "/mnt/ssfs/usr/shared/images")

VIS_URL_REGEX = r"^https://(staging-)?vis.data.visenze.com/v1/image/original/\w{32}$"
VIS_READ_IMAGE_URL = "https://%svis.data.visenze.com/v1/image/original/%s"


def _get_vis_read_url(env: str, uri: str) -> str:
    return VIS_READ_IMAGE_URL % ("" if env == "production" else "staging-", uri)


def read_vis_atomic(
    im_uri: str,
    cache_folder: str | Path = CACHE_FOLDER,
    environment: str = "production",
    max_retry: int = 3,
) -> bytes:
    """
    Read image bytes from a VIS URL with atomic cache writes and sharded subdirectory storage.

    Lookup order (NFS directory-entry-limit safe):
      1. ``<cache_folder>/<md5>``          — legacy flat layout (read-only fallback)
      2. ``<cache_folder>/<md5[:2]>/<md5>`` — new sharded layout

    New downloads always go into the sharded layout.
    """
    if re.search(VIS_URL_REGEX, im_uri):
        md5 = im_uri.split("/")[-1]
    elif re.search(r"\w{32}", im_uri):
        md5 = im_uri
    else:
        raise ValueError(f"URI given {im_uri} is not valid. It must be a valid VIS URL or a MD5 hash")

    # 1. Legacy flat location (existing cached files)
    flat_path = os.path.join(str(cache_folder), md5)
    if os.path.isfile(flat_path):
        with open(flat_path, "rb") as f:
            return f.read()

    # 2. Sharded location
    shard_dir = os.path.join(str(cache_folder), md5[:2])
    shard_dir_is_new = not os.path.isdir(shard_dir)
    os.makedirs(shard_dir, exist_ok=True)
    if shard_dir_is_new:
        os.chmod(shard_dir, 0o777)
    shard_path = os.path.join(shard_dir, md5)
    if os.path.isfile(shard_path):
        with open(shard_path, "rb") as f:
            return f.read()

    # 3. Download into sharded location
    vis_url = _get_vis_read_url(environment, md5)
    retry = 0
    while retry <= max_retry:
        tmp_fd = None
        tmp_name = None
        try:
            with urllib.request.urlopen(vis_url, timeout=30) as resp:  # nosec B310
                raw = resp.read()
            with tempfile.NamedTemporaryFile(dir=shard_dir, delete=False) as tmp_fd:
                tmp_name = tmp_fd.name
                tmp_fd.write(raw)
            os.chmod(tmp_name, 0o666)
            os.replace(tmp_name, shard_path)
            return raw
        except Exception:
            if tmp_name and os.path.exists(tmp_name):
                os.remove(tmp_name)
            retry += 1
            if retry > max_retry:
                raise
            time.sleep(0.1)
    raise RuntimeError(f"Failed to download {vis_url} after {max_retry} retries")

"""
Image Processing Helpers
"""

# CACHE_FOLDER is defined at the top of the file

ASOS_IMAGE_DOMAINS = ("images.asos-media.com", "asos.com")


def _get_image_request_headers(im_url: str) -> dict[str, str]:
    """Return request headers for image download.

    ASOS image hosts can reject non-browser requests; use browser-like headers.
    """
    if any(domain in im_url for domain in ASOS_IMAGE_DOMAINS):
        return {
            "accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;"
                "q=0.8,application/signed-exchange;v=b3;q=0.7"
            ),
            "accept-language": "en-GB,en-US;q=0.9,en;q=0.8,zh-CN;q=0.7,zh-TW;q=0.6,zh;q=0.5",
            "user-agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/143.0.0.0 Safari/537.36"
            ),
            "ngrok-skip-browser-warning": "69420"
        }
    return {}


def _im_url_to_bytes(im_url: str, req_session: Optional[requests.Session]) -> str:
    """Convert image URL to base64 encoded string."""
    try:
        data = BytesIO(read_vis_atomic(im_url))
    except Exception as _:
        headers = _get_image_request_headers(im_url)
        if req_session is None:
            response = requests.get(im_url, headers=headers)
        else:
            response = req_session.get(im_url, headers=headers)
        response.raise_for_status()
        data = BytesIO(response.content)
    return image_to_base64(data)


def im_url_to_base64(im_url: str) -> str:
    return _im_url_to_bytes(im_url, None)


def im_url_to_base64_with_session(im_url: str, req_session: requests.Session) -> str:
    return _im_url_to_bytes(im_url, req_session)


# Function to convert the image to base64 (as requested by OpenAI)
def image_to_base64(image_data, max_short_dimension=768, max_long_dimension=2000) -> str:
    # Resize the image
    resized_generic_image = resize_image(image_data, max_short_dimension, max_long_dimension)

    # Convert resized image to base64
    generic_buffered = BytesIO()
    resized_generic_image.save(generic_buffered, format="JPEG")  # You can change format if necessary
    return base64.b64encode(generic_buffered.getvalue()).decode("utf-8")


# Resize the Image
# Function to resize the image while maintaining aspect ratio
def resize_image(image_data, max_short_dimension=768, max_long_dimension=2000):
    generic_image = Image.open(image_data)
    generic_image = generic_image.convert("RGB")
    generic_width, generic_height = generic_image.size

    if generic_width > generic_height:
        scaling_factor = min(max_long_dimension / generic_width, max_short_dimension / generic_height)
    else:
        scaling_factor = min(max_short_dimension / generic_width, max_long_dimension / generic_height)

    new_width = int(generic_width * scaling_factor)
    new_height = int(generic_height * scaling_factor)

    generic_image = generic_image.resize((new_width, new_height))

    return generic_image


def read_image_from_vis(input_vis: str):
    data = BytesIO(read_vis_atomic(input_vis, cache_folder=CACHE_FOLDER))
    image = np.array(Image.open(data).convert("RGB"))
    return image


def read_image_from_url(input_url: str):
    headers = _get_image_request_headers(input_url)
    image = np.array(Image.open(requests.get(input_url, stream=True, headers=headers).raw).convert("RGB"))
    return image


from curl_cffi import requests as cffi_requests
from typing import Optional

ASOS_HEADERS = {
    "Referer": "https://www.asos.com/",
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
    ),
    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
}

def get_image_content(
    im_url: str | bytes,
    timeout: int = 10,
) -> Optional[bytes]:
    """
    Fetch an image from a URL and return the content as bytes.
    Returns None if fetching or encoding fails.
    """

    # if content is a URL
    if isinstance(im_url, str) and im_url.startswith("http"):
        try:
            if "asos-media.com" in im_url or "asos.com" in im_url:
                # Akamai fingerprints TLS/HTTP2, not just headers —
                # curl_cffi impersonates real Chrome to get past it.
                resp = cffi_requests.get(
                    im_url,
                    impersonate="chrome124",
                    headers=ASOS_HEADERS,
                    timeout=timeout,
                )
            else:
                headers = _get_image_request_headers(im_url)
                resp = requests.get(im_url, timeout=timeout, headers=headers)

            resp.raise_for_status()
            content = resp.content
        except Exception as e:
            logger.error("Failed to fetch/encode image {}: {}", im_url, e)
            return None

    # local file path
    elif isinstance(im_url, str):
        local_path = Path(im_url)
        if not local_path.exists():
            logger.error("Invalid image URL or content: {}", im_url)
            return None
        try:
            content = local_path.read_bytes()
        except Exception as e:
            logger.error("Failed to read local image {}: {}", im_url, e)
            return None

    # if content is already bytes (e.g., from cache)
    elif isinstance(im_url, bytes):
        content = im_url
    else:
        logger.error("Invalid image URL or content: {}", im_url)
        return None

    if not check_image_content_valid(content):
        logger.error("Invalid image content for URL: {}", im_url)
        return None
    return content


def check_image_content_valid(content: bytes) -> bool:
    # try read as image to validate content
    try:
        Image.open(BytesIO(content)).verify()
    except Exception:
        return False

    return True


def get_image_height_and_width(image_content: bytes) -> tuple[int, int] | None:
    """Get (width, height) of an image from its raw bytes."""
    try:
        im = Image.open(BytesIO(image_content))
        return im.width, im.height
    except Exception as e:
        logger.error("Failed to get image dimensions: {}", e)
        return None


class ImageContentCache:
    def __init__(self, cache_dir: str = CACHE_FOLDER):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.makedirs_p()
        self.external_cache_dir = self.cache_dir / "external"
        self.external_cache_dir.makedirs_p()

    def _url_to_path(self, url: str) -> Path:
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.external_cache_dir / url_hash

    def get(self, url: str) -> Optional[bytes]:
        # local file source
        local_path = Path(url)
        if local_path.exists():
            try:
                content = local_path.read_bytes()
                if check_image_content_valid(content):
                    return content
            except Exception as e:
                logger.error("Failed to read local image {}: {}", url, e)
            return None

        # try read from VIS first
        if "vis.data.visenze.com" in url:
            try:
                return read_vis_atomic(url, cache_folder=str(self.cache_dir))
            except Exception as e:
                logger.error("Failed to read VIS image {}: {}", url, e)
                return None

        cache_path = self._url_to_path(url)
        if cache_path.exists():
            content = cache_path.read_bytes()
            if check_image_content_valid(content):
                return content
        content = get_image_content(url)
        if content is not None:
            cache_path.write_bytes(content)
        return content

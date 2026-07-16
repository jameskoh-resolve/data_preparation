def resolve_model_alias(alias: str) -> str:
    alias_lower = alias.lower()
    if alias_lower in ("nano",):
        return "gpt-4.1-nano"
    elif alias_lower in ("mini",):
        return "gpt-4.1-mini"
    elif alias_lower in ("5.4-mini",):
        return "gpt-5.4-mini"
    # If not a known alias, assume it's already a full model name
    return alias

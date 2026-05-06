"""Paradigm registry with @register decorator."""

from core.base_paradigm import BaseParadigm

_REGISTRY: dict[str, type[BaseParadigm]] = {}


def register(cls: type[BaseParadigm]) -> type[BaseParadigm]:
    """Class decorator to register a paradigm."""
    _REGISTRY[cls.short_name] = cls
    return cls


def get_all_paradigms() -> list[BaseParadigm]:
    """Return instances of all registered paradigms."""
    # Trigger imports so @register decorators run
    import paradigms  # noqa: F401
    return [cls() for cls in _REGISTRY.values()]


def get_available_paradigms() -> list[BaseParadigm]:
    """Return instances of registered paradigms that are available."""
    return [p for p in get_all_paradigms() if p.is_available()]

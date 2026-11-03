"""Small typed registries for datasets, models, transforms, and components."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

from cnn_classifier.core.errors import RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    """A name-to-factory registry with explicit duplicate and lookup errors."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Return a decorator that registers an item under ``name``."""
        cleaned = name.strip()
        if not cleaned:
            raise RegistryError(f"{self._name} registry cannot register an empty name")
        if cleaned in self._items:
            raise RegistryError(f"{cleaned!r} is already registered in {self._name}")

        def decorator(item: T) -> T:
            self._items[cleaned] = item
            return item

        return decorator

    def get(self, name: str) -> T:
        """Return the item registered under ``name``."""
        try:
            return self._items[name]
        except KeyError as exc:
            available = ", ".join(self._items) or "<none>"
            raise RegistryError(
                f"{name!r} is not registered in {self._name}; available: {available}"
            ) from exc

    def names(self) -> tuple[str, ...]:
        """Return registered names in insertion order."""
        return tuple(self._items)

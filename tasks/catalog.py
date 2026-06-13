"""Task catalog shared by miners, validators, and task servers.

Add task modules here when introducing a new task kind. Each module must
export ``handler() -> TaskHandler``.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable

from .framework.base import TaskHandler
from .framework.registry import TaskRegistry


# Keep this list small and explicit so task support changes are easy to review.
TASK_MODULES: tuple[str, ...] = (
    ".sn101",
    ".arithmetic",
)

DEFAULT_TASK_KIND = "sn101.tags.v1"


def build_task_registry(
    *,
    modules: Iterable[str] = TASK_MODULES,
    extra_modules: Iterable[str] = (),
    override_modules: Iterable[str] = (),
    default_kind: str | None = DEFAULT_TASK_KIND,
) -> TaskRegistry:
    registry = TaskRegistry(default_kind=default_kind)
    for module_name in [*modules, *extra_modules]:
        if not module_name:
            continue
        registry.register(load_task_handler(module_name))
    for module_name in override_modules:
        if not module_name:
            continue
        registry.register(load_task_handler(module_name), replace=True)
    if default_kind is not None:
        registry.default_handler()
    return registry


def load_task_handler(module_name: str) -> TaskHandler:
    module = importlib.import_module(module_name, package=__package__)
    handler_factory = getattr(module, "handler", None)
    if handler_factory is None:
        raise RuntimeError(f"task module {module_name!r} must export handler()")
    handler = handler_factory()
    if not isinstance(handler, TaskHandler):
        raise TypeError(f"task module {module_name!r} returned {type(handler).__name__}")
    return handler

"""Runtime registry for task-specific lease builders, solvers, and scorers."""

from __future__ import annotations

from .base import TaskHandler


class TaskRegistry:
    def __init__(self, *, default_kind: str | None = None):
        self._handlers: dict[str, TaskHandler] = {}
        self.default_kind = default_kind

    def register(
        self,
        handler: TaskHandler,
        *,
        default: bool = False,
        replace: bool = False,
    ) -> None:
        if handler.kind in self._handlers and not replace:
            raise ValueError(f"task kind {handler.kind!r} is already registered")
        self._handlers[handler.kind] = handler
        if default or self.default_kind is None:
            self.default_kind = handler.kind

    def handler_for(self, kind: str) -> TaskHandler:
        try:
            return self._handlers[kind]
        except KeyError as exc:
            known = ", ".join(sorted(self._handlers)) or "<none>"
            raise KeyError(f"unknown task kind {kind!r}; registered kinds: {known}") from exc

    def default_handler(self) -> TaskHandler:
        if self.default_kind is None:
            raise RuntimeError("no default task kind registered")
        return self.handler_for(self.default_kind)

    def available_kinds(self) -> list[str]:
        return sorted(self._handlers)

    def describe(self) -> list[dict[str, object]]:
        return [
            {
                "kind": handler.kind,
                "spec_version": handler.spec_version,
                "description": handler.description,
                "default": handler.kind == self.default_kind,
            }
            for handler in sorted(self._handlers.values(), key=lambda item: item.kind)
        ]


def default_registry() -> TaskRegistry:
    from ..catalog import build_task_registry

    return build_task_registry()

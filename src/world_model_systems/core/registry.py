from __future__ import annotations

from collections.abc import Iterable

from .spec import ModelAdapter

_REGISTRY: dict[str, ModelAdapter] = {}
_BUILTINS_LOADED = False


def register_model(adapter: ModelAdapter, *, replace: bool = False) -> None:
    model_id = adapter.spec.id
    if model_id in _REGISTRY and not replace:
        raise KeyError(f"Model adapter already registered: {model_id}")
    _REGISTRY[model_id] = adapter


def _load_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from world_model_systems.models.hy_worldplay import HYWorldPlayAdapter
    from world_model_systems.models.lingbot_world_v2 import LingBotWorldV2Adapter
    from world_model_systems.models.matrix_game import MatrixGameAdapter

    for adapter in (MatrixGameAdapter(), LingBotWorldV2Adapter(), HYWorldPlayAdapter()):
        register_model(adapter, replace=True)
    _BUILTINS_LOADED = True


def get_model(model_id: str) -> ModelAdapter:
    _load_builtins()
    try:
        return _REGISTRY[model_id]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown model '{model_id}'. Available: {available}") from exc


def list_models() -> tuple[ModelAdapter, ...]:
    _load_builtins()
    return tuple(_REGISTRY[key] for key in sorted(_REGISTRY))


def iter_model_ids() -> Iterable[str]:
    return (adapter.spec.id for adapter in list_models())

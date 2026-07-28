from __future__ import annotations

from dataclasses import dataclass

from .spec import ModelSpec, OptimizationSpec


@dataclass(frozen=True)
class OptimizationPlan:
    model_id: str
    objective: str
    passes: tuple[OptimizationSpec, ...]
    skipped: tuple[tuple[str, str], ...] = ()

    def ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.passes)


class OptimizationPlanner:
    """Builds a deterministic, dependency-safe optimization plan.

    The planner is deliberately conservative: only implemented passes are selected
    automatically. Experimental passes require ``include_experimental=True``.
    """

    def build(
        self,
        model: ModelSpec,
        *,
        objective: str = "latency",
        requested: tuple[str, ...] | None = None,
        include_experimental: bool = False,
    ) -> OptimizationPlan:
        by_id = {item.id: item for item in model.optimizations}
        candidates = list(requested or tuple(by_id))
        selected: list[OptimizationSpec] = []
        skipped: list[tuple[str, str]] = []
        visiting: set[str] = set()
        done: set[str] = set()

        def add(pass_id: str) -> None:
            if pass_id in done:
                return
            if pass_id in visiting:
                raise ValueError(f"Optimization dependency cycle at {pass_id}")
            item = by_id.get(pass_id)
            if item is None:
                raise KeyError(f"Model {model.id} has no optimization '{pass_id}'")
            if objective not in item.objective:
                skipped.append((pass_id, f"not tagged for objective={objective}"))
                done.add(pass_id)
                return
            if not item.implemented:
                skipped.append((pass_id, "not implemented"))
                done.add(pass_id)
                return
            if item.experimental and not include_experimental:
                skipped.append((pass_id, "experimental"))
                done.add(pass_id)
                return
            missing_caps = set(item.capabilities) - set(model.capabilities)
            if missing_caps:
                names = ", ".join(sorted(cap.value for cap in missing_caps))
                skipped.append((pass_id, f"missing capabilities: {names}"))
                done.add(pass_id)
                return
            visiting.add(pass_id)
            for dependency in item.requires:
                add(dependency)
            selected_ids = {entry.id for entry in selected}
            conflict = selected_ids.intersection(item.conflicts)
            if conflict:
                skipped.append((pass_id, f"conflicts with {sorted(conflict)}"))
            else:
                selected.append(item)
            visiting.remove(pass_id)
            done.add(pass_id)

        for pass_id in candidates:
            add(pass_id)
        return OptimizationPlan(model.id, objective, tuple(selected), tuple(skipped))

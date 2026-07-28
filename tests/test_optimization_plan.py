from world_model_systems import get_model
from world_model_systems.core.plan import OptimizationPlanner


def test_default_plan_excludes_experimental_passes():
    model = get_model("lingbot-world-v2").spec
    plan = OptimizationPlanner().build(model, objective="latency")
    assert "fused-adaln" in plan.ids()
    assert "fp32-causal-rope" not in plan.ids()
    assert ("fp32-causal-rope", "experimental") in plan.skipped


def test_requested_experimental_pass_is_selectable():
    model = get_model("lingbot-world-v2").spec
    plan = OptimizationPlanner().build(
        model,
        objective="latency",
        requested=("fp32-causal-rope",),
        include_experimental=True,
    )
    assert plan.ids() == ("fp32-causal-rope",)

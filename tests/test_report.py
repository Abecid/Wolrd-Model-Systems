import json

from matrixgame_systems.profiling.report import build_summary, render_markdown


def test_report_builds_and_refuses_mismatched_comparison(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "command.json").write_text(
        json.dumps(
            {
                "wall_time_s": 2,
                "frames": 10,
                "num_gpus": 1,
                "return_code": 0,
                "workload": {"seed": 1},
                "workload_fingerprint": "candidate",
            }
        )
    )
    (run / "phases.rank0.json").write_text(
        json.dumps(
            {
                "events": [
                    {"name": "dit_forward", "rank": 0, "wall_ms": 100, "cuda_ms": 90}
                ]
            }
        )
    )
    summary = build_summary(run)
    baseline = dict(summary)
    baseline["workload_fingerprint"] = "baseline"
    markdown = render_markdown(summary, baseline)
    assert "Comparison refused" in markdown
    assert (run / "summary.json").exists()

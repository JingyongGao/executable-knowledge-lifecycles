"""Validate frozen v0.2.6-exp.1 artifacts and regenerate paper data."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return json.loads((ROOT / "artifacts" / name).read_text(encoding="utf-8"))


def main() -> None:
    execution = _read("1c_evaluation.json")
    compilation = _read("1a_1b_metrics.json")
    stress = _read("parameter_sensitivity.json")

    assert execution["spec_version"] == "v0.2.6-exp.1-final"
    for result in execution["results"]:
        vector = result["evaluation_vector"]
        assert vector["r_ood_relative_improvement"] <= 1.0
        assert vector["c_infer_ratio"] == 1.0

    assert compilation["unique_semantic_cases"] == 18
    assert compilation["decode_replicates_per_case"] == {
        "core_preregistered": 30,
        "cross_domain_extension": 20,
    }
    assert compilation["total_decoding_events"] == 390
    conformance = compilation["execution_conformance_1b"]
    assert conformance["semantic_hash_match_rate"] >= 0.95
    assert conformance["execution_vector_matches"] == conformance["semantic_hash_matches"]
    assert compilation["experiment_passed"]

    low = stress["confidence_monotonicity"]
    assert low["low_confidence_grid"] == [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2]
    assert all(
        point["delta_e_ood_absolute_paired_ci95"]["n"] >= 30
        for point in low["low_confidence_wrong_k"]
    )
    assert stress["experiment_passed"]

    for script in ("generate_paper_metrics.py", "build_release_bundle.py"):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script)],
            cwd=ROOT,
            check=True,
        )
    print("v0.2.6-exp.1-final artifacts and paper data validated")


if __name__ == "__main__":
    main()

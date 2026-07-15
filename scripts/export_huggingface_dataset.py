#!/usr/bin/env python3
"""Export the frozen 1A/1B benchmark for the Hugging Face Dataset repository."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from agent_dynamics.benchmark_cases import CASES, LONG_CONTEXT_NOISE


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "huggingface" / "dataset"
FROZEN_FILES = (
    ROOT / "schemas" / "causal_claim.schema.json",
    ROOT / "artifacts" / "1a_1b_metrics.json",
    ROOT / "artifacts" / "1c_evaluation.json",
    ROOT / "artifacts" / "parameter_sensitivity.json",
)
DECODE_REPLICATES = {
    "core_preregistered": 30,
    "cross_domain_extension": 20,
}


def export(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        rows.append(
            {
                "case_id": case.name,
                "domain": case.domain,
                "cohort": case.cohort,
                "source_text": case.source_text,
                "gold_ir": case.gold,
                "critical_paths": [".".join(path) for path in case.critical_paths],
                "decode_replicates": DECODE_REPLICATES[case.cohort],
                "perturbation_context": LONG_CONTEXT_NOISE,
            }
        )

    benchmark_path = output_dir / "benchmark.jsonl"
    benchmark_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    for source in FROZEN_FILES:
        shutil.copyfile(source, output_dir / source.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export(args.output)


if __name__ == "__main__":
    main()

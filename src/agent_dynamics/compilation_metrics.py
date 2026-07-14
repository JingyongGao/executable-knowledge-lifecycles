"""Experiment 1A/1B preregistered evaluation runner.

Group 1 reads natural language directly and returns an untyped prediction object.
Group 2 compiles the same source into the typed causal-claim IR before execution.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .benchmark_cases import CASES, LONG_CONTEXT_NOISE, BenchmarkCase
from .compiler import (
    CompilationError,
    KnowledgeCompiler,
    OllamaBackend,
    envelope_hash,
    semantic_hash,
)
from .experiment import evaluate_ir_execution


REQUIRED_PATHS = (
    ("id",),
    ("cause",),
    ("effect",),
    ("scope",),
    ("scope", "market_phase"),
    ("time_window",),
    ("time_window", "valid_from"),
    ("time_window", "valid_to"),
    ("effect_spec",),
    ("effect_spec", "estimand"),
    ("effect_spec", "estimate"),
    ("effect_spec", "uncertainty"),
    ("claim_confidence",),
    ("provenance",),
    ("version",),
    ("status",),
)


def _at(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _has_path(value: Mapping[str, Any], path: tuple[str, ...]) -> bool:
    current: Any = value
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return True


class DirectNaturalLanguageReader:
    """Group-1 baseline: direct prediction, with JSON syntax but no typed schema."""

    SYSTEM_PROMPT = """Read the provided context and natural-language claim directly,
then predict the fields needed by a 1C consumer. Return JSON only, with keys cause,
effect, lag_order, valid_to, status, claim_confidence, and provenance. Do not use a
typed causal-claim schema; resolve the text yourself."""

    def __init__(self, backend: OllamaBackend) -> None:
        self.backend = backend

    def predict(self, source_text: str, seed_decode: int) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": "LONG_CONTEXT:\n" + LONG_CONTEXT_NOISE},
            {"role": "user", "content": "CLAIM:\n" + source_text},
        ]
        body = {
            "model": self.backend.model,
            "stream": False,
            "format": "json",
            "messages": messages,
            "options": {
                "seed": seed_decode,
                "temperature": self.backend.temperature,
            },
        }
        request = urllib.request.Request(
            self.backend.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.backend.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        decoded = json.loads(payload["message"]["content"])
        if not isinstance(decoded, dict):
            raise ValueError("direct reader did not return an object")
        # Adapt the direct flat prediction to paths shared with typed IR scoring.
        return {
            **decoded,
            "time_window": {
                "lag_order": decoded.get("lag_order"),
                "valid_to": decoded.get("valid_to"),
            },
        }


def _severe_error(prediction: Mapping[str, Any] | None, case: BenchmarkCase) -> bool:
    return prediction is None or any(
        _at(prediction, path) != _at(case.gold, path) for path in case.critical_paths
    )


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def run_metrics(
    *,
    model: str = "qwen-local:latest",
    seed_start: int = 155921,
    decode_samples: int = 30,
    cross_domain_decode_samples: int | None = None,
    temperature: float = 0.2,
) -> dict[str, Any]:
    if decode_samples <= 0 or (
        cross_domain_decode_samples is not None and cross_domain_decode_samples <= 0
    ):
        raise ValueError("decode sample counts must be positive")

    cross_replicates = (
        decode_samples
        if cross_domain_decode_samples is None
        else cross_domain_decode_samples
    )
    cohort_replicates = {
        "core_preregistered": decode_samples,
        "cross_domain_extension": cross_replicates,
    }
    seeds = list(range(seed_start, seed_start + max(cohort_replicates.values())))
    ollama = OllamaBackend(model=model, temperature=temperature, context_noise=LONG_CONTEXT_NOISE)
    compiler = KnowledgeCompiler(ollama)
    direct = DirectNaturalLanguageReader(ollama)

    group1_errors = 0
    group2_errors = 0
    group2_compilation_failures = 0
    inversion_count = 0
    provenance_hallucinations = 0
    required_present = 0
    total_events = sum(cohort_replicates[case.cohort] for case in CASES)
    total_required = len(REQUIRED_PATHS) * total_events
    semantic_hash_matches = 0
    envelope_hash_matches = 0
    execution_matches = 0
    case_counts: dict[str, dict[str, int]] = {
        case.name: {"group_1_errors": 0, "group_2_errors": 0, "compile_failures": 0}
        for case in CASES
    }
    cohort_counts: dict[str, dict[str, int]] = {}
    for case in CASES:
        cohort_counts.setdefault(
            case.cohort,
            {"unique_cases": 0, "events": 0, "group_1_errors": 0, "group_2_errors": 0},
        )
        cohort_counts[case.cohort]["unique_cases"] += 1
        cohort_counts[case.cohort]["events"] += cohort_replicates[case.cohort]
    execution_cache: dict[str, dict[str, float]] = {}

    for replicate_index, seed in enumerate(seeds):
        for case in CASES:
            if replicate_index >= cohort_replicates[case.cohort]:
                continue
            try:
                direct_prediction = direct.predict(case.source_text, seed)
            except Exception:
                direct_prediction = None
            if _severe_error(direct_prediction, case):
                group1_errors += 1
                case_counts[case.name]["group_1_errors"] += 1
                cohort_counts[case.cohort]["group_1_errors"] += 1

            try:
                compiled = compiler.compile(case.source_text, seed_decode=seed)
            except CompilationError:
                compiled = None
                group2_compilation_failures += 1
                case_counts[case.name]["compile_failures"] += 1

            if _severe_error(compiled, case):
                group2_errors += 1
                case_counts[case.name]["group_2_errors"] += 1
                cohort_counts[case.cohort]["group_2_errors"] += 1

            if compiled is None:
                continue
            # JSON null is a present, intentional compiler-stage confidence value.
            required_present += sum(_has_path(compiled, path) for path in REQUIRED_PATHS)
            if case.name == "A_causal_direction" and (
                compiled.get("cause") == case.gold["effect"]
                and compiled.get("effect") == case.gold["cause"]
            ):
                inversion_count += 1
            if compiled.get("provenance") != "user_supplied:natural_language":
                provenance_hallucinations += 1

            # 1B: compare compiled NL IR with the independently frozen native JSON.
            if envelope_hash(compiled) == envelope_hash(case.gold):
                envelope_hash_matches += 1
            if semantic_hash(compiled) == semantic_hash(case.gold):
                semantic_hash_matches += 1
                digest = semantic_hash(compiled)
                if digest not in execution_cache:
                    execution_cache[digest] = evaluate_ir_execution(compiled, epochs=60)
                compiled_vector = execution_cache[digest]
                native_vector = evaluate_ir_execution(deepcopy(case.gold), epochs=60)
                if (
                    compiled_vector["e_inv"] == native_vector["e_inv"]
                    and compiled_vector["e_sens"] == native_vector["e_sens"]
                ):
                    execution_matches += 1

    field_recall = _safe_ratio(required_present, total_required)
    # Compiler rejects unknown fields, so accepted-field precision is 1.0.
    field_precision = 1.0
    field_f1 = _safe_ratio(2 * field_precision * field_recall, field_precision + field_recall)
    group1_rate = _safe_ratio(group1_errors, total_events)
    group2_rate = _safe_ratio(group2_errors, total_events)
    absolute_margin = group1_rate - group2_rate
    relative_reduction = _safe_ratio(absolute_margin, group1_rate)
    redline_passed = absolute_margin >= 0.10 and relative_reduction >= 0.25
    conformance_rate = _safe_ratio(execution_matches, semantic_hash_matches)

    return {
        "experiment": "1A_1B",
        "model": model,
        "decode_seeds": {
            cohort: list(range(seed_start, seed_start + replicates))
            for cohort, replicates in cohort_replicates.items()
        },
        "unique_semantic_cases": len(CASES),
        "decode_replicates_per_case": cohort_replicates,
        "total_decoding_events": total_events,
        "preregistered_failure_bounds": {
            "typed_absolute_error_reduction_min": 0.10,
            "typed_relative_error_reduction_min": 0.25,
            "execution_conformance_required": 1.0,
        },
        "group_1_direct_nl": {
            "severe_semantic_errors": group1_errors,
            "severe_semantic_error_rate": group1_rate,
        },
        "group_2_typed_ir": {
            "compilation_failures": group2_compilation_failures,
            "severe_semantic_errors": group2_errors,
            "severe_semantic_error_rate": group2_rate,
            "field_precision": field_precision,
            "required_field_recall": field_recall,
            "field_f1_score": field_f1,
            "field_corruption_rate": 1.0 - field_f1,
            "causal_inversions": inversion_count,
            "causal_inversion_rate": _safe_ratio(inversion_count, decode_samples),
            "provenance_hallucinations": provenance_hallucinations,
            "provenance_hallucination_rate": _safe_ratio(
                provenance_hallucinations, total_events
            ),
        },
        "confusion_matrix": {
            "rows": ["Group_1_direct_NL", "Group_2_typed_IR"],
            "columns": ["semantically_correct", "severe_semantic_error"],
            "values": [
                [total_events - group1_errors, group1_errors],
                [total_events - group2_errors, group2_errors],
            ],
        },
        "case_breakdown": case_counts,
        "cohort_breakdown": cohort_counts,
        "engineering_advantage_redline": {
            "absolute_error_reduction": absolute_margin,
            "relative_error_reduction": relative_reduction,
            "passed": redline_passed,
        },
        "execution_conformance_1b": {
            "semantic_hash_matches": semantic_hash_matches,
            "semantic_hash_match_rate": _safe_ratio(semantic_hash_matches, total_events),
            "envelope_hash_matches": envelope_hash_matches,
            "envelope_hash_match_rate": _safe_ratio(envelope_hash_matches, total_events),
            "execution_vector_matches": execution_matches,
            "conditional_conformance_rate": conformance_rate,
            "passed": semantic_hash_matches == total_events and conformance_rate == 1.0,
        },
        "experiment_passed": (
            redline_passed
            and semantic_hash_matches == total_events
            and conformance_rate == 1.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen-local:latest")
    parser.add_argument("--seed-start", type=int, default=155921)
    parser.add_argument("--decode-samples", type=int, default=30)
    parser.add_argument("--cross-domain-decode-samples", type=int)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()
    report = run_metrics(
        model=args.model,
        seed_start=args.seed_start,
        decode_samples=args.decode_samples,
        cross_domain_decode_samples=args.cross_domain_decode_samples,
        temperature=args.temperature,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=args.indent)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

"""Generate LaTeX tables/macros from frozen experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from agent_dynamics.benchmark_cases import CASES


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def _read(name: str) -> dict:
    return json.loads((ROOT / "artifacts" / name).read_text(encoding="utf-8"))


def _escape(value: object) -> str:
    text = str(value)
    for source, target in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ):
        text = text.replace(source, target)
    return text


def _fmt(value: float) -> str:
    if value != 0.0 and abs(value) < 0.0001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def main() -> None:
    compilation = _read("1a_1b_metrics.json")
    execution = _read("1c_evaluation.json")
    stress = _read("parameter_sensitivity.json")
    PAPER.mkdir(parents=True, exist_ok=True)

    conformance = compilation["execution_conformance_1b"]
    core = compilation["cohort_breakdown"]["core_preregistered"]
    extension = compilation["cohort_breakdown"]["cross_domain_extension"]
    macros = "\n".join(
        [
            "% Generated; do not edit by hand.",
            rf"\newcommand{{\UniqueCases}}{{{compilation['unique_semantic_cases']}}}",
            rf"\newcommand{{\CoreCases}}{{{core['unique_cases']}}}",
            rf"\newcommand{{\ExtensionCases}}{{{extension['unique_cases']}}}",
            rf"\newcommand{{\TotalEvents}}{{{compilation['total_decoding_events']}}}",
            rf"\newcommand{{\CoreReplicates}}{{{compilation['decode_replicates_per_case']['core_preregistered']}}}",
            rf"\newcommand{{\ExtensionReplicates}}{{{compilation['decode_replicates_per_case']['cross_domain_extension']}}}",
            rf"\newcommand{{\SemanticMatches}}{{{conformance['semantic_hash_matches']}}}",
            rf"\newcommand{{\SemanticMatchPct}}{{{100 * conformance['semantic_hash_match_rate']:.1f}\%}}",
            rf"\newcommand{{\EnvelopeMatchPct}}{{{100 * conformance['envelope_hash_match_rate']:.1f}\%}}",
            rf"\newcommand{{\DirectErrorPct}}{{{100 * compilation['group_1_direct_nl']['severe_semantic_error_rate']:.1f}\%}}",
            rf"\newcommand{{\TypedErrorPct}}{{{100 * compilation['group_2_typed_ir']['severe_semantic_error_rate']:.1f}\%}}",
            rf"\newcommand{{\CoreDirectErrorPct}}{{{100 * core['group_1_errors'] / core['events']:.1f}\%}}",
            rf"\newcommand{{\ExtensionDirectErrorPct}}{{{100 * extension['group_1_errors'] / extension['events']:.1f}\%}}",
            rf"\newcommand{{\SelectedLambda}}{{{stress['lambda_sensitivity']['validation_selected_lambda']}}}",
        ]
    )
    (PAPER / "generated_metrics.tex").write_text(macros + "\n", encoding="utf-8")

    rows = []
    for result in execution["results"]:
        vector = result["evaluation_vector"]
        rows.append(
            " & ".join(
                [
                    _escape(result["group"].replace("Group 0 (", "").replace("Group 1 (", "").replace("Group 2 (", "").replace("Group 3 (", "").replace("Group 4 (", "").replace("Group 5 (", "").rstrip(")")),
                    _fmt(vector["delta_e_ood_absolute"]),
                    _fmt(vector["r_ood_relative_improvement"]),
                    _fmt(vector["e_inv"]),
                    _fmt(vector["e_sens"]),
                    _fmt(vector["c_train_ratio"]),
                    _fmt(vector["c_infer_ratio"]),
                ]
            )
            + r" \\"
        )
    table = r"""% Generated; do not edit by hand.
\begin{table*}[t]
\centering
\caption{Executable-knowledge outcomes. $\Delta E_{\mathrm{OOD}}$ is absolute shortcut-intervention error reduction; $R_{\mathrm{OOD}}$ is its normalized counterpart. Predictive OOD MSE remains separately available in the artifact.}
\label{tab:execution}
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrr}
\toprule
Condition & $\Delta E_{\mathrm{OOD}}$ & $R_{\mathrm{OOD}}$ & $E_{\mathrm{inv}}$ & $E_{\mathrm{sens}}$ & $C_{\mathrm{train}}$ & $C_{\mathrm{infer}}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
}
\end{table*}
"""
    (PAPER / "generated_results_table.tex").write_text(table, encoding="utf-8")

    domains = sorted(
        {
            case.domain: sum(1 for item in CASES if item.domain == case.domain)
            for case in CASES
        }.items()
    )
    domain_rows = "\n".join(
        f"{_escape(domain)} & {count} \\\\" for domain, count in domains
    )
    domain_table = r"""% Generated content; embedded by generated_release.tex.
\begin{tabular}{lr}
\toprule
Domain & Cases \\
\midrule
""" + domain_rows + r"""
\bottomrule
\end{tabular}
"""
    (PAPER / "generated_domain_table.tex").write_text(domain_table, encoding="utf-8")


if __name__ == "__main__":
    main()

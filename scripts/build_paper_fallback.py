"""Build the frozen short paper through a Unicode-safe HTML/PDF fallback.

This builder is used only when no TeX engine is available. It reads the same
three frozen JSON artifacts used by the LaTeX table generator and writes the
canonical release filename, ``paper/main.pdf``.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

from agent_dynamics.benchmark_cases import CASES
ROOT = Path(__file__).resolve().parents[1]


def _read_artifact(name: str) -> dict:
    return json.loads((ROOT / "artifacts" / name).read_text(encoding="utf-8"))


def _fmt(value: float) -> str:
    if value != 0.0 and abs(value) < 0.0001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _condition_name(group: str) -> str:
    return group.split("(", maxsplit=1)[1].rsplit(")", maxsplit=1)[0]


def _artifact_hashes() -> list[tuple[str, str]]:
    rows = []
    for name in (
        "1a_1b_metrics.json",
        "1c_evaluation.json",
        "parameter_sensitivity.json",
    ):
        payload = (ROOT / "artifacts" / name).read_bytes()
        rows.append((name, hashlib.sha256(payload).hexdigest()))
    return rows


def build_html() -> str:
    compilation = _read_artifact("1a_1b_metrics.json")
    execution = _read_artifact("1c_evaluation.json")
    stress = _read_artifact("parameter_sensitivity.json")
    release = json.loads(
        (ROOT / "artifacts" / "release_manifest.json").read_text(encoding="utf-8")
    )

    conformance = compilation["execution_conformance_1b"]
    core = compilation["cohort_breakdown"]["core_preregistered"]
    extension = compilation["cohort_breakdown"]["cross_domain_extension"]
    unique_cases = compilation["unique_semantic_cases"]
    total_events = compilation["total_decoding_events"]
    core_reps = compilation["decode_replicates_per_case"]["core_preregistered"]
    extension_reps = compilation["decode_replicates_per_case"][
        "cross_domain_extension"
    ]
    direct_error = 100 * compilation["group_1_direct_nl"][
        "severe_semantic_error_rate"
    ]
    typed_error = 100 * compilation["group_2_typed_ir"][
        "severe_semantic_error_rate"
    ]
    core_direct_error = 100 * core["group_1_errors"] / core["events"]
    extension_direct_error = (
        100 * extension["group_1_errors"] / extension["events"]
    )
    semantic_pct = 100 * conformance["semantic_hash_match_rate"]
    envelope_pct = 100 * conformance["envelope_hash_match_rate"]
    selected_lambda = stress["lambda_sensitivity"]["validation_selected_lambda"]
    paired_runs = stress["statistical_registration"]["paired_runs"]

    result_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(_condition_name(row['group']))}</td>"
        f"<td>{_fmt(row['evaluation_vector']['delta_e_ood_absolute'])}</td>"
        f"<td>{_fmt(row['evaluation_vector']['r_ood_relative_improvement'])}</td>"
        f"<td>{_fmt(row['evaluation_vector']['e_inv'])}</td>"
        f"<td>{_fmt(row['evaluation_vector']['e_sens'])}</td>"
        f"<td>{_fmt(row['evaluation_vector']['c_train_ratio'])}</td>"
        f"<td>{_fmt(row['evaluation_vector']['c_infer_ratio'])}</td>"
        "</tr>"
        for row in execution["results"]
    )
    domain_counts = {
        case.domain: sum(1 for item in CASES if item.domain == case.domain)
        for case in CASES
    }
    domain_rows = "\n".join(
        f"<tr><td>{html.escape(domain)}</td><td>{count}</td></tr>"
        for domain, count in sorted(domain_counts.items())
    )
    provenance_rows = "\n".join(
        f"<tr><td>{html.escape(name)}</td><td><code>{digest}</code></td></tr>"
        for name, digest in _artifact_hashes()
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Executable Knowledge Lifecycles as Dynamical Control in Agent Systems</title>
<style>
@page {{
  size: A4;
  margin: 18mm 17mm 18mm 17mm;
  @top-left {{
    content: "v0.2.6-exp.1-final | Executable Knowledge Lifecycles";
    color: #475569;
    font-size: 8pt;
  }}
  @top-right {{
    content: "PREPRINT";
    color: #475569;
    font-size: 8pt;
    font-weight: 700;
  }}
  @bottom-center {{
    content: counter(page);
    color: #475569;
    font-size: 8pt;
  }}
}}
* {{ box-sizing: border-box; }}
html {{ font-family: "STIX Two Text", "Times New Roman", "Songti SC", serif; }}
body {{ margin: 0; color: #172033; font-size: 9.5pt; line-height: 1.42; }}
h1 {{ string-set: shorttitle content(); margin: 0 0 3mm; font-size: 20pt; line-height: 1.15; text-align: center; color: #10233f; }}
.subtitle {{ text-align: center; font-size: 12pt; color: #334155; margin-bottom: 4mm; }}
.meta {{ text-align: center; color: #5b6576; font-size: 8.5pt; margin-bottom: 6mm; }}
.abstract {{ border-top: 1.5pt solid #214d76; border-bottom: .5pt solid #94a3b8; padding: 3mm 2mm; background: #f8fafc; margin-bottom: 5mm; }}
.abstract strong {{ font-variant: small-caps; color: #153b60; }}
h2 {{ font-size: 13pt; margin: 5mm 0 2mm; color: #153b60; border-bottom: .4pt solid #cbd5e1; padding-bottom: 1mm; break-after: avoid; }}
h3 {{ font-size: 10.5pt; margin: 3mm 0 1mm; color: #243f5d; break-after: avoid; }}
p {{ margin: 0 0 2.2mm; text-align: justify; hyphens: auto; }}
.equation {{ margin: 3mm auto; padding: 2.5mm 3mm; width: 92%; text-align: center; background: #f6f8fb; border-left: 2pt solid #38658d; font-family: "STIX Two Math", "Times New Roman", serif; font-size: 10pt; break-inside: avoid; }}
code {{ font-family: "SFMono-Regular", "DejaVu Sans Mono", monospace; font-size: 7.1pt; overflow-wrap: anywhere; }}
table {{ width: 100%; border-collapse: collapse; margin: 2mm 0 4mm; table-layout: fixed; break-inside: avoid; }}
caption {{ text-align: left; font-weight: 700; color: #243f5d; margin-bottom: 1.5mm; }}
th {{ border-top: 1.2pt solid #1f2937; border-bottom: .7pt solid #64748b; padding: 1.4mm 1mm; background: #edf2f7; font-size: 7.2pt; line-height: 1.2; }}
td {{ border-bottom: .35pt solid #cbd5e1; padding: 1.25mm 1mm; text-align: right; font-size: 7.35pt; overflow-wrap: anywhere; }}
th:first-child, td:first-child {{ text-align: left; width: 27%; }}
.domains {{ width: 58%; margin-left: auto; margin-right: auto; }}
.domains th:first-child, .domains td:first-child {{ width: 80%; }}
.provenance th:first-child, .provenance td:first-child {{ width: 28%; }}
.note {{ font-size: 8.2pt; color: #475569; }}
.references p {{ padding-left: 5mm; text-indent: -5mm; text-align: left; }}
a {{ color: #174f7a; text-decoration: none; }}
</style>
</head>
<body>
<h1>Executable Knowledge Lifecycles as Dynamical Control in Agent Systems</h1>
<div class="subtitle">A Typed-Compilation Study in Synthetic Causal Environments</div>
<div class="meta">Anonymous technical report &nbsp;|&nbsp; Short paper / preprint &nbsp;|&nbsp; v0.2.6-exp.1-final</div>

<div class="abstract"><strong>Abstract.</strong>
In a synthetic causal environment, a model-external typed compiler separates candidate knowledge from trusted knowledge. Compiler output is constrained to an unscored <code>CANDIDATE</code>; only an epistemic validation plane may assign continuous confidence and admit execution. Across controlled interventions, executing correct knowledge improves robustness, executing incorrect knowledge causes directional harm, and isolating expired knowledge has no effect. Forced activation of expired knowledge restores the harm. These results support a narrow systems claim: an executable knowledge lifecycle acts as a dynamical control structure in an agent system. They do not establish causal discovery or broad real-world validity. The semantic study contains {unique_cases} unique cases ({core['unique_cases']} preregistered core cases and {extension['unique_cases']} cross-domain extensions) and {total_events} decoding events; decode repetitions are reported as repetitions, not independent semantic samples.
</div>

<h2>1. Problem Statement</h2>
<p>Agent systems commonly pass natural-language "knowledge" directly into planning or learning components. This collapses parsing, validation, admission, activation, and execution into one probabilistic channel. We study a stricter lifecycle:</p>
<div class="equation">K<sub>NL</sub> &rarr; <em>compile</em> &rarr; K<sub>IR</sub><sup>candidate</sup> &rarr; <em>validate</em> &rarr; K<sub>IR</sub><sup>trusted</sup> &rarr; <em>activate</em> &rarr; L(K).</div>
<p>The central question is not whether a language model "knows" a causal fact. It is whether system boundaries cause correct, wrong, stale, and adversarially injected knowledge to produce distinguishable and auditable dynamics.</p>

<h2>2. System and Threat Model</h2>
<p>The compiler emits one JSON object conforming to a Draft-07 causal-claim schema. Markdown, missing fields, unknown fields, invalid control metadata, and prefixed or suffixed prose fail outside the model. Governance metadata is sealed as <code>status=CANDIDATE</code> and <code>claim_confidence=null</code>. An epistemic data plane alone converts lineage count, sample size, and consistency into a numeric confidence.</p>
<p>We separate two hashes. <em>SemanticHash</em> covers cause, effect, scope, time window, intervention, and effect specification. <em>EnvelopeHash</em> covers identifier, version, status, provenance, and confidence. This prevents governance serialization from being confused with executable causal semantics.</p>
<p>The threat model includes causal-direction inversion, temporal drift, indirect prompt injection, hallucinated provenance, conflicting claims, expired-claim activation bypass, and ungrounded defaults induced by long-context distractors.</p>

<h2>3. Executable Causal Regularization</h2>
<p>For model f<sub>&theta;</sub>, active claim K, non-causal feature set N(K), and validated confidence c<sub>K</sub>, training minimizes:</p>
<div class="equation">L(K) = L<sub>task</sub> + &lambda; I[K active] c<sub>K</sub> E<sub>x</sub>[ &Sigma;<sub>j in N(K)</sub> || &part;f<sub>&theta;</sub>(x) / &part;x<sub>j</sub> ||<sub>2</sub><sup>2</sup> ].</div>
<p>The synthetic environment contains one structural feature X<sub>1</sub> and two training-environment shortcuts X<sub>2</sub>, X<sub>3</sub>. Shortcut correlations reverse out of distribution.</p>
<p>We define E<sub>OOD</sub> operationally as the held-out shortcut-intervention MSE with X<sub>1</sub> fixed. It is the same raw observation reported as E<sub>inv</sub>; it is not a general-purpose measure of distribution shift. Raw predictive OOD MSE is retained separately.</p>
<div class="equation">&Delta;E<sub>OOD</sub> = E<sub>base</sub> - E<sub>K</sub>, &nbsp;&nbsp; R<sub>OOD</sub> = &Delta;E<sub>OOD</sub> / (E<sub>base</sub> + 10<sup>-12</sup>).</div>

<h2>4. Experiments</h2>
<h3>Typed compilation (1A)</h3>
<p>The core cohort has {core['unique_cases']} handcrafted semantic cases with {core_reps} decode replicates each. The extension adds {extension['unique_cases']} cases across quantitative trading, advertising technology, clinical and industrial safety, supply chain, cybersecurity, energy, manufacturing, credit, logistics, agriculture, recommendation, telecommunications, education technology, and insurance, with {extension_reps} replicates per case. Thus {total_events} events do not imply {total_events} independent semantic cases.</p>
<p>We preregister a <em>severe semantic error</em> as any decoding event that reverses causal direction, changes required temporal or intervention semantics, fabricates protected provenance, violates frozen governance metadata, or disagrees with Gold IR on a required executable field.</p>
<h3>Execution conformance (1B)</h3>
<p>Compiled natural language and native Gold IR are compared by SemanticHash and EnvelopeHash. When semantic hashes match, downstream (E<sub>inv</sub>, E<sub>sens</sub>) must be exactly equal.</p>
<h3>Knowledge execution (1C)</h3>
<p>No-K, correct, wrong, partial, stale, and fault-activated stale conditions share data, initialization, and decoding seeds.</p>
<h3>Stress tests</h3>
<p>We scan &lambda; in {{0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10}} and confidence, including c<sub>K</sub> in {{0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2}}. Every point uses {paired_runs} paired seeds and a paired Student-t 95% interval. Lambda ({selected_lambda}) is selected on validation MSE without inspecting held-out OOD outcomes.</p>

<h2>5. Results</h2>
<p>The direct natural-language path has a severe semantic error rate of {direct_error:.1f}% overall ({core_direct_error:.1f}% in the core cohort and {extension_direct_error:.1f}% in the extension), while typed compilation has {typed_error:.1f}%. SemanticHash agreement is {semantic_pct:.1f}% ({conformance['semantic_hash_matches']}/{total_events}), and EnvelopeHash agreement is {envelope_pct:.1f}%. These are system-level results: deterministic source anchors and model-external normalization are part of the compiler and should not be interpreted as raw language-model accuracy.</p>

<table id="execution"><caption>Table 1. Executable-knowledge outcomes. Delta E<sub>OOD</sub> is absolute shortcut-intervention error reduction; R<sub>OOD</sub> is its normalized counterpart.</caption>
<thead><tr><th>Condition</th><th>&Delta;E<sub>OOD</sub></th><th>R<sub>OOD</sub></th><th>E<sub>inv</sub></th><th>E<sub>sens</sub></th><th>C<sub>train</sub></th><th>C<sub>infer</sub></th></tr></thead>
<tbody>{result_rows}</tbody></table>

<table id="domains" class="domains"><caption>Table 2. Semantic benchmark coverage. Domains are independent case labels, not independent statistical samples.</caption>
<thead><tr><th>Domain</th><th>Cases</th></tr></thead><tbody>{domain_rows}</tbody></table>

<p>As shown in <a href="#execution">Table 1</a>, correct knowledge removes nearly all shortcut-intervention sensitivity. Wrong knowledge amplifies error in the opposite direction. Partial knowledge produces an intermediate outcome. Inactive stale knowledge is exactly equivalent to No-K under shared initialization, while activation bypass produces material harm. Low-confidence wrong knowledge remains harmful but its mean harm increases monotonically with confidence. <a href="#domains">Table 2</a> reports the manually authored domain labels.</p>

<h2>6. Limitations and Validity</h2>
<p>The execution environment is synthetic, linear, and deliberately small. The cross-domain cases broaden vocabulary and business context, but they remain manually authored templates with explicit variable labels. Decode replicates measure stochastic stability, not population-level semantic diversity. The benchmark evaluates compilation fidelity against frozen Gold IR; it does not validate the truth of those claims. E<sub>OOD</sub> approximately equals E<sub>inv</sub> as an operational identity in this experiment, not evidence that invariance universally measures OOD generalization. Confidence scoring is a reference control law rather than a calibrated empirical estimator. No clinical, financial, or safety deployment claim follows from these results.</p>

<h2>7. Reproducibility and Provenance</h2>
<p>Schemas, compiler, loss operator, frozen cases, seeds, raw metrics, confidence intervals, and failure flags are versioned together. Paper tables are generated directly from JSON artifacts. The compilation artifact records case counts and decode-replicate counts separately. All model inference was local; untrusted raw generations are not persisted in the formal artifact.</p>
<p>The release locator status is <code>{release['source_control']['status']}</code>. Repository URL, tag, commit, and external bundle URL remain explicitly unassigned because the freeze source is not a Git worktree; no synthetic locator is substituted. The deterministic evidence bundle is <code>{release['bundle']['path']}</code> with SHA-256 <code>{release['bundle']['sha256']}</code>.</p>
<table class="provenance"><caption>Table 3. SHA-256 digests of the frozen evidence artifacts used for this build.</caption>
<thead><tr><th>Artifact</th><th>SHA-256</th></tr></thead><tbody>{provenance_rows}</tbody></table>

<h2>8. Conclusion</h2>
<p>Within the stated synthetic scope, the evidence supports a control-structure interpretation of executable knowledge. Correct execution improves behavior; wrong execution damages it; lifecycle isolation prevents stale knowledge from acting; bypassing isolation restores harm. The engineering result is the separation of probabilistic parsing from deterministic admission and activation, not a claim of autonomous causal understanding.</p>

<h2>References</h2>
<div class="references">
<p>[1] J. Pearl. <em>Causality: Models, Reasoning, and Inference</em>, 2nd ed. Cambridge University Press, 2009.</p>
<p>[2] J. Peters, D. Janzing, and B. Sch&ouml;lkopf. <em>Elements of Causal Inference</em>. MIT Press, 2017.</p>
<p>[3] M. Arjovsky et al. "Invariant Risk Minimization." arXiv:1907.02893, 2019.</p>
</div>
<p class="note">Release build: v0.2.6-exp.1-final. This document is generated from frozen local artifacts and contains no externally supplied performance claims.</p>
</body>
</html>
"""


def build_reportlab_pdf(output: Path) -> str:
    """Pure-Python fallback when WeasyPrint's native Pango stack is absent."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        KeepTogether,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    compilation = _read_artifact("1a_1b_metrics.json")
    execution = _read_artifact("1c_evaluation.json")
    stress = _read_artifact("parameter_sensitivity.json")
    release = json.loads(
        (ROOT / "artifacts" / "release_manifest.json").read_text(encoding="utf-8")
    )
    conformance = compilation["execution_conformance_1b"]
    core = compilation["cohort_breakdown"]["core_preregistered"]
    extension = compilation["cohort_breakdown"]["cross_domain_extension"]

    font_candidates = (
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/Supplemental/Songti.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    font_path = next((path for path in font_candidates if path.exists()), None)
    body_font = "Times-Roman"
    bold_font = "Times-Bold"
    if font_path is not None:
        try:
            pdfmetrics.registerFont(TTFont("ReleaseUnicode", str(font_path)))
            body_font = "ReleaseUnicode"
            bold_font = "ReleaseUnicode"
            if pdfmetrics.stringWidth("中文", body_font, 9) <= 0:
                raise RuntimeError("CJK font smoke test returned zero width")
        except Exception as error:
            print(f"[WARNING] Unicode font registration failed: {error}")
            font_path = None

    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=19 * mm,
        bottomMargin=18 * mm,
        title="Executable Knowledge Lifecycles as Dynamical Control in Agent Systems",
        author="Anonymous technical report",
        subject="v0.2.6-exp.1-final",
    )
    sheet = getSampleStyleSheet()
    title = ParagraphStyle(
        "ReleaseTitle",
        parent=sheet["Title"],
        fontName=bold_font,
        fontSize=19,
        leading=22,
        textColor=colors.HexColor("#10233f"),
        alignment=TA_CENTER,
        spaceAfter=5,
    )
    subtitle = ParagraphStyle(
        "ReleaseSubtitle",
        parent=sheet["Normal"],
        fontName=body_font,
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#334155"),
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    body = ParagraphStyle(
        "ReleaseBody",
        parent=sheet["BodyText"],
        fontName=body_font,
        fontSize=9.2,
        leading=12.4,
        textColor=colors.HexColor("#172033"),
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    heading = ParagraphStyle(
        "ReleaseHeading",
        parent=sheet["Heading2"],
        fontName=bold_font,
        fontSize=12.5,
        leading=15,
        textColor=colors.HexColor("#153b60"),
        spaceBefore=10,
        spaceAfter=5,
    )
    subheading = ParagraphStyle(
        "ReleaseSubheading",
        parent=sheet["Heading3"],
        fontName=bold_font,
        fontSize=10,
        leading=12,
        textColor=colors.HexColor("#243f5d"),
        spaceBefore=5,
        spaceAfter=2,
    )
    abstract_style = ParagraphStyle(
        "ReleaseAbstract",
        parent=body,
        backColor=colors.HexColor("#f5f8fb"),
        borderColor=colors.HexColor("#38658d"),
        borderWidth=0.8,
        borderPadding=8,
        spaceAfter=10,
    )
    equation = ParagraphStyle(
        "ReleaseEquation",
        parent=body,
        alignment=TA_CENTER,
        backColor=colors.HexColor("#f6f8fb"),
        borderColor=colors.HexColor("#38658d"),
        borderWidth=0.5,
        borderPadding=6,
        leftIndent=12,
        rightIndent=12,
        spaceBefore=3,
        spaceAfter=7,
    )
    small = ParagraphStyle(
        "ReleaseSmall",
        parent=body,
        fontSize=7.3,
        leading=9.2,
        alignment=TA_LEFT,
    )

    story = [
        Paragraph("Executable Knowledge Lifecycles as Dynamical Control in Agent Systems", title),
        Paragraph("A Typed-Compilation Study in Synthetic Causal Environments", subtitle),
        Paragraph(
            "Anonymous technical report | Short paper / preprint | v0.2.6-exp.1-final",
            subtitle,
        ),
        Paragraph(
            "<b>Abstract.</b> In a synthetic causal environment, a model-external typed "
            "compiler separates candidate knowledge from trusted knowledge. Compiler output "
            "is constrained to an unscored <font name='Courier'>CANDIDATE</font>; only an "
            "epistemic validation plane may assign continuous confidence and admit execution. "
            "Across controlled interventions, executing correct knowledge improves robustness, "
            "executing incorrect knowledge causes directional harm, and isolating expired "
            "knowledge has no effect. Forced activation of expired knowledge restores the harm. "
            "These results support a narrow systems claim: an executable knowledge lifecycle "
            "acts as a dynamical control structure in an agent system. They do not establish "
            "causal discovery or broad real-world validity. The semantic study contains "
            f"{compilation['unique_semantic_cases']} unique cases ({core['unique_cases']} "
            f"preregistered core cases and {extension['unique_cases']} cross-domain extensions) "
            f"and {compilation['total_decoding_events']} decoding events; decode repetitions are "
            "reported as repetitions, not independent semantic samples.",
            abstract_style,
        ),
        Paragraph("1. Problem Statement", heading),
        Paragraph(
            'Agent systems commonly pass natural-language "knowledge" directly into planning '
            "or learning components. This collapses parsing, validation, admission, activation, "
            "and execution into one probabilistic channel. We study a stricter lifecycle:", body
        ),
        Paragraph(
            "K<sub>NL</sub> -> compile -> K<sub>IR</sub><super>candidate</super> -> validate -> "
            "K<sub>IR</sub><super>trusted</super> -> activate -> L(K)", equation
        ),
        Paragraph(
            'The central question is not whether a language model "knows" a causal fact. It is '
            "whether system boundaries cause correct, wrong, stale, and adversarially injected "
            "knowledge to produce distinguishable and auditable dynamics.", body
        ),
        Paragraph("2. System and Threat Model", heading),
        Paragraph(
            "The compiler emits one JSON object conforming to a Draft-07 causal-claim schema. "
            "Markdown, missing fields, unknown fields, invalid control metadata, and prefixed or "
            "suffixed prose fail outside the model. Governance metadata is sealed as "
            "<font name='Courier'>status=CANDIDATE</font> and "
            "<font name='Courier'>claim_confidence=null</font>. An epistemic data plane alone "
            "converts lineage count, sample size, and consistency into a numeric confidence.", body
        ),
        Paragraph(
            "We separate two hashes. <i>SemanticHash</i> covers cause, effect, scope, time "
            "window, intervention, and effect specification. <i>EnvelopeHash</i> covers "
            "identifier, version, status, provenance, and confidence. This prevents governance "
            "serialization from being confused with executable causal semantics.", body
        ),
        Paragraph(
            "The threat model includes causal-direction inversion, temporal drift, indirect "
            "prompt injection, hallucinated provenance, conflicting claims, expired-claim "
            "activation bypass, and ungrounded defaults induced by long-context distractors.", body
        ),
        Paragraph("3. Executable Causal Regularization", heading),
        Paragraph(
            "For model f(theta), active claim K, non-causal feature set N(K), and validated "
            "confidence c(K), training minimizes:", body
        ),
        Paragraph(
            "L(K) = L(task) + lambda I[K active] c(K) E[x][ sum(j in N(K)) "
            "|| df(x) / dx(j) ||<super>2</super> ]", equation
        ),
        Paragraph(
            "The synthetic environment contains one structural feature X1 and two "
            "training-environment shortcuts X2 and X3. Shortcut correlations reverse out of "
            "distribution.", body
        ),
        Paragraph(
            "We define E(OOD) operationally as the held-out shortcut-intervention MSE with X1 "
            "fixed. It is the same raw observation reported as E(inv); it is not a "
            "general-purpose measure of distribution shift. Raw predictive OOD MSE is retained "
            "separately.", body
        ),
        Paragraph(
            "Delta E(OOD) = E(base) - E(K),    R(OOD) = Delta E(OOD) / "
            "(E(base) + 10<super>-12</super>)", equation
        ),
        Paragraph("4. Experiments", heading),
        Paragraph("Typed compilation (1A)", subheading),
        Paragraph(
            f"The core cohort has {core['unique_cases']} handcrafted semantic cases with "
            f"{compilation['decode_replicates_per_case']['core_preregistered']} decode "
            f"replicates each. The extension adds {extension['unique_cases']} cases across "
            "fifteen business and safety domains, with "
            f"{compilation['decode_replicates_per_case']['cross_domain_extension']} replicates "
            f"per case. Thus {compilation['total_decoding_events']} events do not imply "
            f"{compilation['total_decoding_events']} independent semantic cases.", body
        ),
        Paragraph("Execution conformance (1B)", subheading),
        Paragraph(
            "Compiled natural language and native Gold IR are compared by SemanticHash and "
            "EnvelopeHash. When semantic hashes match, downstream (E(inv), E(sens)) must be "
            "exactly equal.", body
        ),
        Paragraph(
            "We preregister a <i>severe semantic error</i> as any decoding event that "
            "reverses causal direction, changes required temporal or intervention semantics, "
            "fabricates protected provenance, violates frozen governance metadata, or "
            "disagrees with Gold IR on a required executable field.", body
        ),
        Paragraph("Knowledge execution (1C)", subheading),
        Paragraph(
            "No-K, correct, wrong, partial, stale, and fault-activated stale conditions share "
            "data, initialization, and decoding seeds.", body
        ),
        Paragraph("Stress tests", subheading),
        Paragraph(
            "We scan lambda in {0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10} and confidence, "
            "including c(K) in {0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2}. Every point uses "
            f"{stress['statistical_registration']['paired_runs']} paired seeds and a paired "
            "Student-t 95% interval. Lambda is selected on validation MSE without inspecting "
            "held-out OOD outcomes.", body
        ),
        Paragraph("5. Results", heading),
        Paragraph(
            "The direct natural-language path has a severe semantic error rate of "
            f"{100 * compilation['group_1_direct_nl']['severe_semantic_error_rate']:.1f}% "
            f"overall ({100 * core['group_1_errors'] / core['events']:.1f}% in the core cohort "
            f"and {100 * extension['group_1_errors'] / extension['events']:.1f}% in the "
            "extension), while typed compilation has "
            f"{100 * compilation['group_2_typed_ir']['severe_semantic_error_rate']:.1f}%. "
            f"SemanticHash agreement is {100 * conformance['semantic_hash_match_rate']:.1f}% "
            f"({conformance['semantic_hash_matches']}/{compilation['total_decoding_events']}), "
            f"and EnvelopeHash agreement is {100 * conformance['envelope_hash_match_rate']:.1f}%. "
            "These are system-level results: deterministic source anchors and model-external "
            "normalization are part of the compiler and should not be interpreted as raw "
            "language-model accuracy.", body
        ),
    ]

    header = [
        Paragraph("Condition", small),
        Paragraph("Delta E(OOD)", small),
        Paragraph("R(OOD)", small),
        Paragraph("E(inv)", small),
        Paragraph("E(sens)", small),
        Paragraph("C(train)", small),
        Paragraph("C(infer)", small),
    ]
    rows = [header]
    for row in execution["results"]:
        vector = row["evaluation_vector"]
        rows.append(
            [
                Paragraph(html.escape(_condition_name(row["group"])), small),
                _fmt(vector["delta_e_ood_absolute"]),
                _fmt(vector["r_ood_relative_improvement"]),
                _fmt(vector["e_inv"]),
                _fmt(vector["e_sens"]),
                _fmt(vector["c_train_ratio"]),
                _fmt(vector["c_infer_ratio"]),
            ]
        )
    result_table = Table(rows, colWidths=[103, 53, 48, 46, 48, 53, 50], repeatRows=1)
    result_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#243f5d")),
                ("FONTNAME", (0, 0), (-1, -1), body_font),
                ("FONTSIZE", (0, 1), (-1, -1), 6.9),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LINEABOVE", (0, 0), (-1, 0), 1, colors.HexColor("#1f2937")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#64748b")),
                ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
            ]
        )
    )
    story.extend(
        [
            Paragraph(
                "Table 1. Executable-knowledge outcomes. Delta E(OOD) is absolute "
                "shortcut-intervention error reduction; R(OOD) is its normalized counterpart.",
                small,
            ),
            result_table,
            Spacer(1, 8),
        ]
    )

    domain_counts = {
        case.domain: sum(1 for item in CASES if item.domain == case.domain)
        for case in CASES
    }
    domain_data = [[Paragraph("Domain", small), Paragraph("Cases", small)]] + [
        [domain, str(count)] for domain, count in sorted(domain_counts.items())
    ]
    domain_table = Table(domain_data, colWidths=[245, 55], repeatRows=1)
    domain_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef4")),
                ("FONTNAME", (0, 0), (-1, -1), body_font),
                ("FONTSIZE", (0, 0), (-1, -1), 7.2),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
            ]
        )
    )
    story.extend(
        [
            KeepTogether(
                [
                    Paragraph(
                        "Table 2. Semantic benchmark coverage. Domains are independent case "
                        "labels, not independent statistical samples.", small
                    ),
                    domain_table,
                ]
            ),
            Paragraph(
                "As shown in Table 1, correct knowledge removes nearly all "
                "shortcut-intervention sensitivity. Wrong knowledge amplifies error in the "
                "opposite direction. Partial knowledge produces an intermediate outcome. "
                "Inactive stale knowledge is exactly equivalent to No-K under shared "
                "initialization, while activation bypass produces material harm. Low-confidence "
                "wrong knowledge remains harmful but its mean harm increases monotonically with "
                "confidence. Table 2 reports the manually authored domain labels.", body
            ),
            Paragraph("6. Limitations and Validity", heading),
            Paragraph(
                "The execution environment is synthetic, linear, and deliberately small. The "
                "cross-domain cases broaden vocabulary and business context, but they remain "
                "manually authored templates with explicit variable labels. Decode replicates "
                "measure stochastic stability, not population-level semantic diversity. The "
                "benchmark evaluates compilation fidelity against frozen Gold IR; it does not "
                "validate the truth of those claims. E(OOD) approximately equals E(inv) as an "
                "operational identity in this experiment, not evidence that invariance "
                "universally measures OOD generalization. Confidence scoring is a reference "
                "control law rather than a calibrated empirical estimator. No clinical, "
                "financial, or safety deployment claim follows from these results.", body
            ),
            Paragraph("7. Reproducibility and Provenance", heading),
            Paragraph(
                "Schemas, compiler, loss operator, frozen cases, seeds, raw metrics, confidence "
                "intervals, and failure flags are versioned together. Paper tables are generated "
                "directly from JSON artifacts. The compilation artifact records case counts and "
                "decode-replicate counts separately. All model inference was local; untrusted raw "
                "generations are not persisted in the formal artifact.", body
            ),
            Paragraph(
                f"Release locator status: {release['source_control']['status']}. Repository "
                "URL, tag, commit, and external bundle URL remain explicitly unassigned because "
                "the freeze source is not a Git worktree; no synthetic locator is substituted. "
                f"Evidence bundle: {release['bundle']['path']}; SHA-256: "
                f"{release['bundle']['sha256']}.", small
            ),
        ]
    )
    provenance_data = [[Paragraph("Artifact", small), Paragraph("SHA-256", small)]] + [
        [
            Paragraph(html.escape(name), small),
            Paragraph(f"<font name='Courier' size='6'>{digest}</font>", small),
        ]
        for name, digest in _artifact_hashes()
    ]
    provenance_table = Table(provenance_data, colWidths=[135, 310], repeatRows=1)
    provenance_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef4")),
                ("FONTNAME", (0, 0), (-1, -1), body_font),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
            ]
        )
    )
    story.extend(
        [
            Paragraph(
                "Table 3. SHA-256 digests of the frozen evidence artifacts used for this build.",
                small,
            ),
            provenance_table,
            Paragraph("8. Conclusion", heading),
            Paragraph(
                "Within the stated synthetic scope, the evidence supports a control-structure "
                "interpretation of executable knowledge. Correct execution improves behavior; "
                "wrong execution damages it; lifecycle isolation prevents stale knowledge from "
                "acting; bypassing isolation restores harm. The engineering result is the "
                "separation of probabilistic parsing from deterministic admission and activation, "
                "not a claim of autonomous causal understanding.", body
            ),
            Paragraph("References", heading),
            Paragraph(
                "[1] J. Pearl. <i>Causality: Models, Reasoning, and Inference</i>, 2nd ed. "
                "Cambridge University Press, 2009.", body
            ),
            Paragraph(
                "[2] J. Peters, D. Janzing, and B. Scholkopf. <i>Elements of Causal "
                "Inference</i>. MIT Press, 2017.", body
            ),
            Paragraph(
                '[3] M. Arjovsky et al. "Invariant Risk Minimization." arXiv:1907.02893, 2019.',
                body,
            ),
        ]
    )

    def decorate_page(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont(body_font, 7.5)
        canvas.setFillColor(colors.HexColor("#526172"))
        canvas.drawString(16 * mm, A4[1] - 11 * mm, "v0.2.6-exp.1-final | Executable Knowledge Lifecycles")
        canvas.drawRightString(A4[0] - 16 * mm, A4[1] - 11 * mm, "PREPRINT")
        canvas.drawCentredString(A4[0] / 2, 10 * mm, str(doc.page))
        canvas.restoreState()

    document.build(story, onFirstPage=decorate_page, onLaterPages=decorate_page)
    return str(font_path) if font_path else "built-in Latin font only"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "paper" / "main.pdf")
    parser.add_argument("--html", type=Path, default=ROOT / "paper" / "report.html")
    parser.add_argument("--log", type=Path, default=ROOT / "paper" / "main.log")
    args = parser.parse_args()

    markup = build_html()
    args.html.parent.mkdir(parents=True, exist_ok=True)
    args.html.write_text(markup, encoding="utf-8")
    engine = "WeasyPrint HTML-to-PDF fallback"
    font_note = "STIX Two Text, Times New Roman, Songti SC"
    try:
        from weasyprint import HTML

        HTML(string=markup, base_url=str(ROOT)).write_pdf(args.output)
    except (ImportError, OSError) as error:
        print(f"[WARNING] WeasyPrint native stack unavailable: {error}")
        print("[INFO] Switching to pure-Python ReportLab fallback.")
        font_note = build_reportlab_pdf(args.output)
        engine = "ReportLab pure-Python PDF fallback"
    args.log.write_text(
        f"Build engine: {engine}\n"
        "Source encoding: UTF-8\n"
        f"Unicode font fallback: {font_note}\n"
        "LaTeX log checks: not applicable to fallback; static TeX audit follows\n"
        "Overfull hbox: not applicable to CSS paged-media layout\n"
        "Cross-reference mode: internal HTML anchors\n",
        encoding="utf-8",
    )
    print(f"Fallback engine: {engine}")
    print(f"Fallback HTML:   {args.html}")
    print(f"Fallback PDF:    {args.output}")


if __name__ == "__main__":
    main()

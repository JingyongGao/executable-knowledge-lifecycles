"""Fail-closed structural, log, provenance, and PDF audit for the paper build."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from pypdf import PdfReader


REQUIRED_DISCLOSURES = (
    "not independent semantic samples",
    "does not validate the truth",
    "not a general-purpose measure of distribution shift",
    "manually authored templates",
    "status=CANDIDATE",
    "claim_confidence=null",
    "Correct execution improves behavior; wrong execution damages it",
    "severe semantic error",
    "reverses causal direction",
)
FORBIDDEN_PHRASES = (
    "as an ai",
    "as a language model",
    "i cannot",
    "i think",
    "here is your json",
)


def _without_comments(text: str) -> str:
    kept = []
    for line in text.splitlines():
        match = re.search(r"(?<!\\)%", line)
        kept.append(line[: match.start()] if match else line)
    return "\n".join(kept)


def audit_tex_structure(paper_dir: Path) -> None:
    sources = [paper_dir / "main.tex", *sorted(paper_dir.glob("generated_*.tex"))]
    combined = "\n".join(
        _without_comments(path.read_text(encoding="utf-8")) for path in sources
    )

    brace_depth = 0
    escaped = False
    for char in combined:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
            if brace_depth < 0:
                raise AssertionError("closing brace without matching opening brace")
    if brace_depth:
        raise AssertionError(f"unclosed LaTeX brace depth: {brace_depth}")

    stack: list[str] = []
    for match in re.finditer(r"\\(begin|end)\{([^{}]+)\}", combined):
        action, environment = match.groups()
        if action == "begin":
            stack.append(environment)
        elif not stack or stack.pop() != environment:
            raise AssertionError(f"mismatched LaTeX environment: {environment}")
    if stack:
        raise AssertionError(f"unclosed LaTeX environments: {stack}")

    labels = set(re.findall(r"\\label\{([^{}]+)\}", combined))
    references = set(re.findall(r"\\ref\{([^{}]+)\}", combined))
    missing = references - labels
    if missing:
        raise AssertionError(f"undefined static LaTeX references: {sorted(missing)}")
    print(f"[PASS] LaTeX braces/environments closed; {len(references)} references resolved")


def audit_log(log_path: Path) -> None:
    if not log_path.exists():
        raise AssertionError("paper/main.log was not produced")
    log = log_path.read_text(encoding="utf-8", errors="replace")
    overfull = [
        float(value)
        for value in re.findall(
            r"Overfull \\hbox \(([0-9]+(?:\.[0-9]+)?)pt too wide\)", log
        )
    ]
    severe = [value for value in overfull if value > 20.0]
    if severe:
        raise AssertionError(f"severe overfull hbox values exceed 20pt: {severe}")
    lowered = log.lower()
    for marker in (
        "undefined references",
        "there were undefined",
        "citation `",
        "reference `",
    ):
        if marker in lowered and "undefined" in lowered:
            raise AssertionError(f"LaTeX log contains unresolved marker: {marker}")
    for marker in ("fatal error", "emergency stop"):
        if marker in lowered:
            raise AssertionError(f"LaTeX log contains fatal marker: {marker}")
    maximum = max(overfull, default=0.0)
    print(f"[PASS] main.log has no overfull hbox above 20pt (max={maximum:.3f}pt)")
    print("[PASS] main.log has no unresolved reference/citation warning")


def audit_unicode_source(paper_dir: Path) -> None:
    manuscript = (paper_dir / "main.tex").read_text(encoding="utf-8")
    if r"\usepackage[utf8]{inputenc}" not in manuscript:
        raise AssertionError("pdfTeX UTF-8 input declaration missing")
    if r"\usepackage{fontspec}" not in manuscript:
        raise AssertionError("XeTeX/Tectonic Unicode font support missing")
    report = paper_dir / "report.html"
    if report.exists():
        markup = report.read_text(encoding="utf-8")
        if '<meta charset="utf-8">' not in markup or "Songti SC" not in markup:
            raise AssertionError("HTML fallback lacks UTF-8/CJK font declarations")
    print("[PASS] UTF-8 source and XeTeX/HTML Unicode font paths declared")


def audit_disclosures(paper_dir: Path, extracted_text: str) -> None:
    manuscript = (paper_dir / "main.tex").read_text(encoding="utf-8")
    normalized_tex = manuscript.replace(r"\texttt{", "").replace("}", "")
    normalized_tex = normalized_tex.replace(r"\_", "_")
    combined = normalized_tex + "\n" + extracted_text
    for phrase in REQUIRED_DISCLOSURES:
        if phrase.lower() not in combined.lower():
            raise AssertionError(f"required disclosure missing: {phrase}")
    for phrase in FORBIDDEN_PHRASES:
        if phrase in combined.lower():
            raise AssertionError(f"forbidden model-self-description found: {phrase}")
    print("[PASS] permission metadata and disclosure invariants locked")


def audit_release_manifest(paper_dir: Path) -> None:
    root = paper_dir.parent
    manifest_path = root / "artifacts" / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle = root / manifest["bundle"]["path"]
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    if digest != manifest["bundle"]["sha256"]:
        raise AssertionError("release evidence bundle hash does not match manifest")
    source = manifest["source_control"]
    if source["status"] == "externally_locatable" and not all(
        source[key]
        for key in ("repository_url", "commit_hash", "release_tag", "external_bundle_url")
    ):
        raise AssertionError("externally_locatable status requires all release locators")
    print(
        f"[PASS] release manifest/bundle locked: {manifest['bundle']['file_count']} files, "
        f"sha256={digest}, status={source['status']}"
    )


def render_pages(pdf_path: Path, render_dir: Path) -> int:
    if render_dir.exists():
        shutil.rmtree(render_dir)
    render_dir.mkdir(parents=True)
    prefix = render_dir / "main"
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm:
        subprocess.run(
            [pdftoppm, "-png", "-r", "150", str(pdf_path), str(prefix)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    else:
        import fitz

        document = fitz.open(pdf_path)
        matrix = fitz.Matrix(150 / 72, 150 / 72)
        for index, page in enumerate(document, start=1):
            page.get_pixmap(matrix=matrix, alpha=False).save(
                render_dir / f"main-{index:02d}.png"
            )
    pages = len(list(render_dir.glob("main-*.png")))
    if pages == 0:
        raise AssertionError("PDF rendering produced no page images")
    print(f"[PASS] rendered {pages} PDF pages for visual audit: {render_dir}")
    return pages


def audit_pdf(pdf_path: Path, require_tex_producer: bool) -> str:
    if not pdf_path.is_file() or pdf_path.stat().st_size < 20_000:
        raise AssertionError("paper/main.pdf is missing or implausibly small")
    reader = PdfReader(pdf_path)
    producer = str((reader.metadata or {}).get("/Producer", ""))
    if require_tex_producer and any(
        marker in producer.lower() for marker in ("reportlab", "weasyprint")
    ):
        raise AssertionError(f"canonical PDF has non-TeX producer: {producer}")
    if require_tex_producer and not producer:
        raise AssertionError("canonical PDF does not declare a producer")
    if not 3 <= len(reader.pages) <= 12:
        raise AssertionError(f"unexpected short-paper page count: {len(reader.pages)}")
    for index, page in enumerate(reader.pages, start=1):
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        if width <= 0 or height <= 0:
            raise AssertionError(f"invalid media box on page {index}")
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    collapsed = re.sub(r"\s+", " ", extracted)
    for marker in (
        "Executable Knowledge Lifecycles",
        "v0.2.6-exp.1-final",
        "390 decoding events",
        "100.0%",
        "severe semantic error",
        "Evidence bundle",
        "References",
    ):
        if marker.lower() not in collapsed.lower():
            raise AssertionError(f"required PDF text missing: {marker}")
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    print(
        f"[PASS] PDF text/media audit: {len(reader.pages)} pages, "
        f"{pdf_path.stat().st_size} bytes, producer={producer}, sha256={digest}"
    )
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", type=Path, default=Path("paper"))
    parser.add_argument("--render-dir", type=Path, default=Path("tmp/pdfs/main"))
    parser.add_argument("--require-tex-producer", action="store_true")
    args = parser.parse_args()
    paper_dir = args.paper_dir.resolve()
    pdf_path = paper_dir / "main.pdf"

    audit_tex_structure(paper_dir)
    audit_log(paper_dir / "main.log")
    audit_unicode_source(paper_dir)
    extracted = audit_pdf(pdf_path, args.require_tex_producer)
    audit_disclosures(paper_dir, extracted)
    audit_release_manifest(paper_dir)
    render_pages(pdf_path, args.render_dir.resolve())
    print("PAPER AUDIT PASSED")


if __name__ == "__main__":
    main()

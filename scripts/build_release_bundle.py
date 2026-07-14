"""Create a deterministic evidence bundle and fail-closed release manifest."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_ID = "v0.2.6-exp.1-final"
BUNDLE_RELATIVE = Path("artifacts") / f"{RELEASE_ID}-evidence.zip"
MANIFEST_RELATIVE = Path("artifacts") / "release_manifest.json"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _source_control() -> dict[str, str | None]:
    repository_url = os.environ.get("RELEASE_REPOSITORY_URL") or _git(
        "remote", "get-url", "origin"
    )
    commit_hash = os.environ.get("RELEASE_COMMIT_HASH") or _git("rev-parse", "HEAD")
    release_tag = os.environ.get("RELEASE_TAG") or _git(
        "describe", "--tags", "--exact-match"
    )
    external_bundle_url = os.environ.get("RELEASE_BUNDLE_URL")

    if commit_hash is not None and not re.fullmatch(r"[0-9a-fA-F]{40,64}", commit_hash):
        raise ValueError("RELEASE_COMMIT_HASH must be a 40-64 character hex digest")
    if repository_url is not None and not (
        repository_url.startswith(("https://", "ssh://", "git@"))
    ):
        raise ValueError("RELEASE_REPOSITORY_URL must be an HTTPS or Git SSH locator")
    if external_bundle_url is not None and not external_bundle_url.startswith("https://"):
        raise ValueError("RELEASE_BUNDLE_URL must use HTTPS")

    complete = all((repository_url, commit_hash, release_tag, external_bundle_url))
    return {
        "status": "externally_locatable" if complete else "local_freeze_only",
        "repository_url": repository_url,
        "commit_hash": commit_hash,
        "release_tag": release_tag,
        "external_bundle_url": external_bundle_url,
    }


def _release_files() -> list[Path]:
    paths: set[Path] = {
        ROOT / ".gitattributes",
        ROOT / ".gitignore",
        ROOT / "CITATION.cff",
        ROOT / "CONTRIBUTING.md",
        ROOT / "LICENSE",
        ROOT / "LIMITATIONS.md",
        ROOT / "NOTICE",
        ROOT / "README.md",
        ROOT / "RELEASE_NOTES.md",
        ROOT / "SECURITY.md",
        ROOT / "pyproject.toml",
        ROOT / "requirements.txt",
        ROOT / "paper" / "README.md",
        ROOT / "paper" / "Makefile",
        ROOT / "paper" / "main.tex",
    }
    for pattern in (
        "schemas/*.json",
        "src/agent_dynamics/*.py",
        "tests/*.py",
        "scripts/*.py",
        "scripts/*.sh",
        ".github/ISSUE_TEMPLATE/*.yml",
        "paper/generated_metrics.tex",
        "paper/generated_results_table.tex",
        "paper/generated_domain_table.tex",
        "artifacts/1a_1b_metrics.json",
        "artifacts/1c_evaluation.json",
        "artifacts/parameter_sensitivity.json",
    ):
        paths.update(ROOT.glob(pattern))
    return sorted(path for path in paths if path.is_file())


def _write_deterministic_zip(paths: list[Path], destination: Path) -> list[dict]:
    entries = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            payload = path.read_bytes()
            mode = 0o755 if os.access(path, os.X_OK) else 0o644
            info = zipfile.ZipInfo(relative, date_time=(2026, 7, 13, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = mode << 16
            info.create_system = 3
            archive.writestr(info, payload)
            entries.append(
                {"path": relative, "bytes": len(payload), "sha256": _sha256(payload)}
            )
    return entries


def _tex_escape(text: str) -> str:
    for source, target in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ):
        text = text.replace(source, target)
    return text


def _break_hash(value: str) -> str:
    return r"\allowbreak{}".join(value[index : index + 8] for index in range(0, len(value), 8))


def _locator(value: str | None, missing: str) -> str:
    return _tex_escape(value) if value else r"\emph{" + _tex_escape(missing) + "}"


def _url_locator(value: str | None, missing: str) -> str:
    return rf"\url{{{value}}}" if value else r"\emph{" + _tex_escape(missing) + "}"


def _write_release_table(manifest: dict) -> None:
    source = manifest["source_control"]
    bundle = manifest["bundle"]
    repository = _url_locator(
        source["repository_url"], "not assigned; freeze source is not a Git worktree"
    )
    tag = _locator(source["release_tag"], "not assigned")
    commit = _locator(source["commit_hash"], "not available")
    external = _url_locator(source["external_bundle_url"], "not assigned")
    table = rf"""% Generated by scripts/build_release_bundle.py; do not edit.
\begin{{table*}}[t]
\centering
\caption{{Semantic coverage and release provenance. Domain labels are not independent statistical samples; null external locators are disclosed rather than replaced with synthetic repository metadata.}}
\label{{tab:evidence}}
\begin{{minipage}}[t]{{0.43\textwidth}}
\centering
\textbf{{Panel A: benchmark coverage}}\\[0.5ex]
\small
\input{{generated_domain_table.tex}}
\end{{minipage}}
\hfill
\begin{{minipage}}[t]{{0.53\textwidth}}
\centering
\textbf{{Panel B: release and evidence locators}}\\[0.5ex]
\footnotesize
\begin{{tabular}}{{@{{}}p{{0.25\textwidth}}p{{0.70\textwidth}}@{{}}}}
\toprule
Release field & Frozen value \\
\midrule
Release ID & \texttt{{{_tex_escape(manifest['release_id'])}}} \\
Repository URL & {repository} \\
VCS tag & {tag} \\
Source commit & \texttt{{{commit}}} \\
Evidence bundle & \path{{{bundle['path']}}} \\
External bundle URL & {external} \\
Bundle SHA-256 & {{\ttfamily\scriptsize {_break_hash(bundle['sha256'])}}} \\
\bottomrule
\end{{tabular}}
\end{{minipage}}
\end{{table*}}
"""
    (ROOT / "paper" / "generated_release.tex").write_text(table, encoding="utf-8")


def main() -> None:
    paths = _release_files()
    bundle_path = ROOT / BUNDLE_RELATIVE
    entries = _write_deterministic_zip(paths, bundle_path)
    bundle_payload = bundle_path.read_bytes()
    manifest = {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "canonical_manuscript": "paper/main.tex",
        "canonical_pdf": "paper/main.pdf",
        "source_control": _source_control(),
        "bundle": {
            "path": BUNDLE_RELATIVE.as_posix(),
            "bytes": len(bundle_payload),
            "sha256": _sha256(bundle_payload),
            "deterministic_timestamp": "2026-07-13T00:00:00+08:00",
            "file_count": len(entries),
        },
        "files": entries,
    }
    (ROOT / MANIFEST_RELATIVE).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_release_table(manifest)
    print(
        f"release bundle: {BUNDLE_RELATIVE} "
        f"({manifest['bundle']['sha256']}, {len(entries)} files)"
    )
    print(f"source locator status: {manifest['source_control']['status']}")


if __name__ == "__main__":
    main()

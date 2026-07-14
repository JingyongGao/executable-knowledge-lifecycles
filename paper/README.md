# Paper build

Generate all tables and macros from the frozen JSON artifacts:

```bash
.venv/bin/python scripts/generate_paper_metrics.py
```

For release validation plus regeneration, run from the repository root:

```bash
.venv/bin/python scripts/freeze_release.py
```

The release build is intentionally launched from the repository root. It creates a clean
Python 3.12 virtual environment, validates the frozen artifacts, requires the exact frozen
62-test count, builds `paper/main.pdf`, checks LaTeX structure/logs/cross-references, and
renders every page to `tmp/pdfs/main/` for visual inspection:

```bash
scripts/reproduce_and_compile.sh
```

The canonical `paper/main.pdf` must be compiled from `paper/main.tex`. The build prefers
`tectonic`, then `latexmk` with `xelatex`, then direct `xelatex`; the release audit rejects a
ReportLab or WeasyPrint producer for the canonical path. Without a TeX engine,
`ALLOW_NON_TEX_PREVIEW=1` may be used to create `paper/main.preview.pdf`, but that preview
cannot trigger `RELEASE LOCK SUCCESS`.

Tectonic uses the explicit, versioned bundle mirror in the script. Set `TECTONIC_OFFLINE=1`
after the first successful cache fill to require an offline-only build.

`scripts/freeze_release.py` also creates `artifacts/release_manifest.json` and a deterministic
`artifacts/v0.2.6-exp.1-final-evidence.zip`. Repository URL, Git tag, commit hash, and external
bundle URL are accepted only from a real Git worktree or `RELEASE_*` environment variables;
missing locators remain JSON `null` and are disclosed in the paper.
Local source is loaded through a scoped `PYTHONPATH=src`; this deliberately avoids making
release validation depend on an index-hosted copy of the Hatchling build backend.

`generated_metrics.tex`, `generated_results_table.tex`, and
`generated_domain_table.tex` are generated files. The manuscript explicitly distinguishes
unique semantic cases from decode replicates and treats the cross-domain set as a manually
authored robustness extension rather than evidence of unrestricted semantic generalization.

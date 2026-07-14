# Contributing

Contributions that improve reproducibility, find boundary failures, or broaden the
benchmark without overstating independence are welcome.

## Reproduction reports

Run `./scripts/reproduce_and_compile.sh` from a clean checkout. A report should
include the commit, operating system, Python and TeX engine versions, exact command,
test summary, artifact hashes, and the smallest relevant log excerpt. Do not attach
credentials or private model prompts.

## Adversarial semantic cases

New cases should include natural-language input, frozen Gold IR, the threat being
tested, why every executable field is grounded in the text, and whether the case is
independent or a transformation of an existing case. Decode replicates must remain
separate from the count of unique semantic cases.

## Pull requests

Keep changes scoped and add or update tests for every behavioral change. The
required gate is the complete frozen test suite plus successful artifact generation
and paper audit. Generated JSON, release manifest, evidence ZIP, checksums, and PDF
must be refreshed when their sources change.

By contributing, you agree that your contribution is licensed under Apache-2.0.

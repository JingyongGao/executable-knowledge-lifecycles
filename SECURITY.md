# Security policy

## Supported release

Security fixes are currently targeted at `v0.2.6-exp.1-final` and the `main`
branch. Earlier experimental snapshots are not supported.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository. Do not
open a public issue containing working exploit details, private data, credentials,
or unpublished model endpoint information. Include the affected commit, a minimal
reproduction, expected and observed boundary behavior, and impact.

Relevant security boundaries include:

- compiler escape from strict JSON or schema validation;
- `CANDIDATE`/`VALIDATED` privilege escalation;
- unauthorized `claim_confidence` assignment;
- provenance fabrication or envelope-to-semantic behavior smuggling;
- activation-window bypass and stale-claim execution;
- SemanticHash collisions or omitted executable fields;
- secret exposure through model endpoints, logs, fixtures, or release bundles.

This policy is not a promise that the research artifact is suitable for
safety-critical use. See `LIMITATIONS.md` for deployment non-claims.

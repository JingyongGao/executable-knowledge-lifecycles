# Limitations and non-claims

This release supports a narrow engineering claim about executable knowledge
lifecycles. Its evidence must be interpreted within the following boundaries.

- The execution study uses a synthetic, linear, small-scale causal environment.
- The semantic benchmark contains 18 manually authored cases. Domain labels do
  not turn those templates into representative samples of their industries.
- There are 390 decoding events, but only 18 unique semantic cases. Replicate
  decodes measure stability and are not independent semantic samples.
- Deterministic anchors are a system-level compiler capability. They must not be
  attributed to the unaided language model.
- SemanticHash equality shows fidelity to frozen Gold IR on executable fields. It
  does not show that the Gold IR is causally true in the world.
- In the current experiment, `E_OOD` and `E_inv` are two names for the same raw
  held-out shortcut-intervention MSE. They are not independent confirmations.
- Confidence assignment is demonstrated as an architectural separation, not as a
  validated general-purpose epistemic scoring method.
- The results do not justify clinical, financial, industrial, cybersecurity, or
  other safety-critical production deployment.

Useful extensions include externally sourced corpora, blinded annotation,
nonlinear and partially observed environments, independent replications, stronger
baseline models, and preregistered real-domain evaluations with domain experts.

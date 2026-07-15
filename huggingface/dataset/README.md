---
pretty_name: Executable Knowledge Lifecycles Benchmark
license: apache-2.0
language:
- en
- zh
task_categories:
- text-classification
tags:
- agent-systems
- causal-reasoning
- structured-generation
- prompt-injection
- reproducibility
configs:
- config_name: default
  data_files:
  - split: test
    path: benchmark.jsonl
---

# Executable Knowledge Lifecycles Benchmark

This dataset is the frozen semantic benchmark for Experiments 1A and 1B of
**Executable Knowledge Lifecycles as Dynamical Control in Agent Systems**.
It tests whether an untrusted natural-language parser can be placed behind a
typed, model-external compilation boundary without acquiring validation authority.

- Version: `0.2.6-exp.1-final`
- Version DOI: [10.5281/zenodo.21361204](https://doi.org/10.5281/zenodo.21361204)
- Source release: [GitHub v0.2.6-exp.1-final](https://github.com/JingyongGao/executable-knowledge-lifecycles/releases/tag/v0.2.6-exp.1-final)
- Release collection: [Executable Knowledge Lifecycles](https://huggingface.co/collections/jingyongai/executable-knowledge-lifecycles)
- Author: Yongjing Gao

## Dataset structure

`benchmark.jsonl` contains one row per independently authored semantic case:

| Field | Meaning |
|---|---|
| `case_id` | Stable case identifier |
| `domain` | Synthetic application-domain label |
| `cohort` | Preregistered core or cross-domain extension |
| `source_text` | Natural-language input treated as untrusted data |
| `gold_ir` | Frozen human-authored candidate causal IR |
| `critical_paths` | Executable fields used for severe-error scoring |
| `decode_replicates` | Decode seeds used for this case |
| `perturbation_context` | Long-context interference injected during evaluation |

The repository also includes the causal-claim JSON Schema and the three frozen
evaluation JSON files.

## Counts and non-independence disclosure

- Unique semantic cases: **18**
- Preregistered core cases: **3**, with 30 decode replicates per case
- Cross-domain extension cases: **15**, with 20 decode replicates per case
- Total decoding events: **390**

The 390 decoding events are repeated stochastic decodes, **not 390 independent
semantic samples**. The cross-domain cases are manually authored templates rather
than samples from a naturally occurring business distribution.

## Intended use

The benchmark supports:

1. causal-direction, lag, validity-window, and prompt-injection regression tests;
2. strict JSON and required-field evaluation;
3. SemanticHash and EnvelopeHash conformance checks;
4. adversarial additions that preserve the published schema contract.

## Severe semantic error

A severe semantic error is an output that reverses causal direction, changes a
required timing or intervention semantic, fabricates protected provenance, grants
itself validation authority, or otherwise disagrees with the frozen Gold IR on a
case's `critical_paths`.

## Limitations and safety

- Gold IR is a human annotation target; it is not proof that a causal claim is true.
- Deterministic source anchors are a system capability, not a claim about the bare model.
- The study uses synthetic, linear, small-scale environments.
- Domain labels including clinical safety and credit risk do not make these examples
  suitable for medical, financial, industrial, or other safety-critical deployment.
- In the frozen experiment, `E_OOD` and `E_inv` are operationally derived from the
  same intervention-style error measurement and are not independent mechanisms.

See the repository's [LIMITATIONS.md](https://github.com/JingyongGao/executable-knowledge-lifecycles/blob/main/LIMITATIONS.md)
before interpreting the metrics.

## Citation

```bibtex
@software{gao_2026_executable_knowledge_lifecycles,
  author    = {Yongjing Gao},
  title     = {Executable Knowledge Lifecycles as Dynamical Control in Agent Systems},
  version   = {0.2.6-exp.1-final},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21361204},
  url       = {https://doi.org/10.5281/zenodo.21361204}
}
```

# Executable Knowledge Lifecycles as Dynamical Control in Agent Systems

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/release-v0.2.6--exp.1--final-2f6f9f.svg)](https://github.com/JingyongGao/executable-knowledge-lifecycles/releases/tag/v0.2.6-exp.1-final)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21361204.svg)](https://doi.org/10.5281/zenodo.21361204)

This repository is the executable evidence package for a narrow systems claim:
in a synthetic causal environment, a model-external typed compiler can separate
candidate knowledge from trusted knowledge. Executing correct knowledge improves
robustness; executing wrong knowledge causes directional harm; isolating expired
knowledge has no effect; bypassing that isolation restores the harm.

The claim is deliberately limited. This work does **not** establish causal
discovery, real-world validity, or readiness for clinical, financial, industrial,
or safety-critical deployment.

## Frozen empirical result

| Condition | Absolute OOD error reduction | Relative improvement | $E_{inv}$ | $E_{sens}$ |
|---|---:|---:|---:|---:|
| No K | 0.0000 | 0.0000 | 1.6330 | 1.0392 |
| Correct K | 1.6330 | 1.0000 | $6.35\times10^{-8}$ | 0.0008 |
| Wrong K | -4.1422 | -2.5365 | 5.7751 | 2.0116 |
| Partial K | 1.3570 | 0.8310 | 0.2760 | 0.3924 |
| Stale K (isolated) | 0.0000 | 0.0000 | 1.6330 | 1.0392 |
| Stale K (fault activated) | -5.2758 | -3.2308 | 6.9088 | 2.0079 |

These values are frozen observations from the included synthetic experiment, not
population estimates. Exact machine-readable values and definitions live in
[`artifacts/1c_evaluation.json`](artifacts/1c_evaluation.json).

## Evidence and reproducibility

- Permanent Zenodo archive (this release): [10.5281/zenodo.21361204](https://doi.org/10.5281/zenodo.21361204)
- Zenodo concept DOI (all versions): [10.5281/zenodo.21361203](https://doi.org/10.5281/zenodo.21361203)
- Paper: [`paper/main.pdf`](paper/main.pdf)
- Typed-compilation metrics: [`artifacts/1a_1b_metrics.json`](artifacts/1a_1b_metrics.json)
- Execution metrics: [`artifacts/1c_evaluation.json`](artifacts/1c_evaluation.json)
- Parameter sensitivity: [`artifacts/parameter_sensitivity.json`](artifacts/parameter_sensitivity.json)
- Release manifest: [`artifacts/release_manifest.json`](artifacts/release_manifest.json)
- Deterministic evidence bundle: [`artifacts/v0.2.6-exp.1-final-evidence.zip`](artifacts/v0.2.6-exp.1-final-evidence.zip)
- SHA-256 list: [`artifacts/SHA256SUMS`](artifacts/SHA256SUMS)

The semantic benchmark contains 18 manually authored cases: 3 preregistered core
cases with 30 decoding replicates each, plus 15 cross-domain extensions with 20
replicates each. The resulting 390 decoding events are **not** presented as 390
independent semantic samples. SemanticHash measures compilation fidelity to frozen
Gold IR; it does not establish that the Gold IR is causally true.

## One-command reproduction

Prerequisites are Python 3.10+ and either Tectonic or XeLaTeX. Then run:

```bash
./scripts/reproduce_and_compile.sh
```

The script creates a clean virtual environment, regenerates frozen artifacts, runs
the exact 62-test invariant suite, builds `paper/main.pdf` from `paper/main.tex`,
audits the PDF and TeX log, and writes release checksums. To force the cached,
network-free Tectonic path after its bundle has been installed:

```bash
TECTONIC_OFFLINE=1 ./scripts/reproduce_and_compile.sh
```

For the local-model compilation benchmark, set `AGENT_DYNAMICS_MODEL` to an Ollama
model (or point `OLLAMA_CHAT_URL` at a compatible endpoint) and invoke
`run-1a-1b-metrics` as documented below.

## Scope, citation, and governance

Read [`LIMITATIONS.md`](LIMITATIONS.md) before interpreting the numbers. Cite this
release using [`CITATION.cff`](CITATION.cff), report vulnerabilities according to
[`SECURITY.md`](SECURITY.md), and see [`CONTRIBUTING.md`](CONTRIBUTING.md) for
reproduction reports and adversarial benchmark additions. The code and manuscript
sources are licensed under Apache License 2.0; see [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE).

---

## 中文说明

# 智能体系统动力学规约 v0.2.6-exp.1-final

这是规约第一轨与 1C 因果执行实验的可执行参考实现。

## 包含内容

- `schemas/audit_event.schema.json`：外部事件审计元数据契约。
- `schemas/causal_claim.schema.json`：版本化因果声明准入契约。
- `agent_dynamics.losses.CausalRegularizedLoss`：对已声明非因果特征施加输入梯度平方惩罚。
- `agent_dynamics.system`：先审计后解析、历史 Belief 不可变、因果声明顺序版本化，以及规范层禁止倒灌。
- `agent_dynamics.experiment`：共享数据、初始化和解码种子的六组 1C 对照实验。
- `agent_dynamics.compiler`：Ollama/API 可插拔的自然语言到类型化因果 IR 编译器。
- `agent_dynamics.compilation_metrics`：Experiment 1A/1B 的 30-seed 对抗评估器。

## 安装与验证

需要 Python 3.10 或更新版本。推荐使用独立虚拟环境：

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest
```

运行完整实验并输出 JSON：

```bash
.venv/bin/run-1c-experiment
# 或
.venv/bin/python -m agent_dynamics.experiment --epochs 350
```

运行本地模型的 1A/1B 分层对抗评估：

```bash
AGENT_DYNAMICS_MODEL=qwen2.5vl:3b \
  .venv/bin/run-1a-1b-metrics \
  --decode-samples 30 \
  --cross-domain-decode-samples 20 \
  --output artifacts/1a_1b_metrics.json
```

编译器默认连接 `http://127.0.0.1:11434/api/chat`，也可通过
`OLLAMA_CHAT_URL` 指向兼容端点。模型输出会依次经过精确 `json.loads`、字段白名单、
受限控制字段、Draft-07 Schema 和显式源文本锚点校验；Markdown、额外说明、缺字段和
`VALIDATED` 权限走私均无法越过编译边界。
Compiler 固定输出 `status=CANDIDATE` 与 `claim_confidence=null`；只有
`EpistemicDataPlane` 能依据血缘记录、样本量与一致性分数生成连续置信度并转为
`VALIDATED`。

六组设置固定为：No K、Correct K、Wrong K、Partial K、正常 Stale K，以及绕过
Activation 的 fault-injected Stale K。每组输出：

`C(K) = (delta_e_ood_absolute, r_ood_relative_improvement, e_inv, e_sens, c_train_ratio, c_infer_ratio)`

- `e_ood_base_raw/e_ood_group_raw`：固定 X1、置换 X2/X3 后的原始干预 MSE。
- `delta_e_ood_absolute`：`E_base-E_group`，保留无界负值。
- `r_ood_relative_improvement`：绝对下降除以 `E_base+1e-12`，上界为 1。
- `e_inv`：`e_ood_group_raw` 的显式别名。
- `e_sens`：模型对标准化 X1 的梯度与真实结构斜率之间的绝对误差，越低越好。
- `c_train_ratio`：训练墙钟时间比；`c_infer_ratio`：同构前向图成本比。

`wall_seconds` 和 `c_train_ratio` 会因机器负载变化；其余数值由固定种子复现。
仓库中的一次已验证实跑结果保存在 `artifacts/1c_evaluation.json`。
Experiment 1A/1B 的正式结果保存在 `artifacts/1a_1b_metrics.json`。报告分开记录
3 个 preregistered core cases（每例 30 次解码）与 15 个 cross-domain extension cases
（每例 20 次解码），合计 18 个独立语义场景和 390 个 decoding events。

论文源文件位于 `paper/main.tex`。运行
`.venv/bin/python scripts/generate_paper_metrics.py` 可直接从正式 JSON 重建论文表格。
终局可复现构建使用 `scripts/reproduce_and_compile.sh`：脚本重建 `.venv_clean`、安装
`requirements.txt`、执行冻结校验与精确 62 项测试，并通过 TeX 引擎或 Unicode-safe
TeX 路径生成正式资产 `paper/main.pdf`，随后执行日志、交叉引用、披露不变量、PDF
Producer 及逐页渲染审计。无 TeX 时只允许显式生成 `paper/main.preview.pdf`，不能冒充
正式冻结件。冻结入口同时生成确定性证据 ZIP 与 `artifacts/release_manifest.json`；
不存在的仓库 URL、tag 或 commit 均保持为空，不得伪造。

## 参数敏感性与故障注入

运行 N=30 配对种子的 λ、置信度和激活故障扫描：

```bash
.venv/bin/run-parameter-stress \
  --paired-runs 30 \
  --epochs 350 \
  --output artifacts/parameter_sensitivity.json
```

每个参数点报告基于 seed 内配对差值的 Student-t 95% 置信区间。λ 由独立
validation 环境的 MSE 选择，held-out OOD 反转环境不参与选择。一次冻结的正式结果
保存在 `artifacts/parameter_sensitivity.json`。

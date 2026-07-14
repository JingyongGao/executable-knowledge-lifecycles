#!/usr/bin/env bash
# ==============================================================================
# v0.2.6-exp.1-final clean-environment reproduction and paper build
# ==============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== [1/4] 启动沙箱隔离：重建干净的虚拟环境 ==="
rm -rf .venv_clean
if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON="$PYTHON_BIN"
elif command -v python3.12 >/dev/null 2>&1; then
    PYTHON="$(command -v python3.12)"
else
    PYTHON="$(command -v python3)"
fi
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
"$PYTHON" -m venv .venv_clean
source .venv_clean/bin/activate
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import sys; print("isolated_python=", sys.executable); print("version=", sys.version)'

echo "=== [2/4] 重新压实基础依赖与数据冻结入口 ==="
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/freeze_release.py

echo "=== [3/4] 触发回归断言：执行最终 62 项不变量测试 ==="
mkdir -p tmp
pytest tests/ -v 2>&1 | tee tmp/reproduce_pytest.log
if ! grep -Eq '(^|[[:space:]])62 passed([,[:space:]]|$)' tmp/reproduce_pytest.log; then
    echo "[FATAL] 回归计数不是冻结的 62 passed。" >&2
    exit 1
fi
echo "[PASS] 冻结回归计数：62 passed"

echo "=== [4/4] 启动 LaTeX 排版引擎：自动化构建与排版审计 ==="
cd paper
rm -f main.pdf main.aux main.out main.log main.toc
BUILD_ENGINE=""

if command -v tectonic >/dev/null 2>&1; then
    echo "检测到 tectonic 引擎，启动不可信输入隔离编译..."
    TECTONIC_BUNDLE_URL="${TECTONIC_BUNDLE_URL:-https://data1.fullyjustified.net/tlextras-2022.0r0.tar}"
    TECTONIC_FLAGS=(--untrusted --keep-logs --bundle "$TECTONIC_BUNDLE_URL")
    if [[ "${TECTONIC_OFFLINE:-0}" == "1" ]]; then
        TECTONIC_FLAGS+=(--only-cached)
    fi
    tectonic "${TECTONIC_FLAGS[@]}" main.tex
    BUILD_ENGINE="tectonic"
elif command -v latexmk >/dev/null 2>&1 && command -v xelatex >/dev/null 2>&1; then
    echo "检测到 latexmk + xelatex，引擎执行多轮交叉引用对齐..."
    latexmk -xelatex -pdf -interaction=nonstopmode -halt-on-error main.tex
    BUILD_ENGINE="latexmk-xelatex"
elif command -v xelatex >/dev/null 2>&1; then
    echo "检测到 xelatex，引擎执行双轮交叉引用对齐..."
    xelatex -interaction=nonstopmode -halt-on-error main.tex
    xelatex -interaction=nonstopmode -halt-on-error main.tex
    BUILD_ENGINE="xelatex"
else
    if [[ "${ALLOW_NON_TEX_PREVIEW:-0}" == "1" ]]; then
        echo "[WARNING] 无 TeX 引擎；仅生成非正式 main.preview.pdf，不触发 release lock。"
        python ../scripts/build_paper_fallback.py \
            --output main.preview.pdf \
            --html report.html \
            --log main.preview.log
        exit 2
    fi
    echo "[FATAL] 正式 paper/main.pdf 必须由 tectonic/xelatex 从 main.tex 构建。" >&2
    echo "安装 tectonic，或设置 ALLOW_NON_TEX_PREVIEW=1 只生成预览件。" >&2
    exit 1
fi
cd "$ROOT"

python scripts/audit_paper.py \
    --paper-dir paper \
    --render-dir tmp/pdfs/main \
    --require-tex-producer
python scripts/generate_release_checksums.py

echo "=============================================================================="
echo "RELEASE LOCK SUCCESS: 冻结、62 项测试、PDF 构建与审计全部通过。"
echo "engine=$BUILD_ENGINE"
echo "artifact=$ROOT/paper/main.pdf"
echo "checksums=$ROOT/artifacts/SHA256SUMS"
echo "=============================================================================="

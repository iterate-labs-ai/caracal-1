#!/usr/bin/env bash
# Ignite-3B reproducibility bundle.
# Reproduces iterate-labs-ai/ignite-3b-v1 from scratch on Kaggle T4 x2.
#
# Prereqs:
#   HF_TOKEN with write access to iterate-labs-ai/*
#   Kaggle account + kaggle CLI credentials
#   Optional: ANTHROPIC_API_KEY (Cond B baseline only)

set -euo pipefail

REPO="iterate-labs-ai/caracal-1"
BRANCH="s07-hybrid-agentic"
BASE_MODEL="unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
OUT_ROOT="/tmp/ignite_repro"

echo "=== Ignite-3B reproduction pipeline ==="
mkdir -p "$OUT_ROOT"

if [ ! -d caracal-1 ]; then
    git clone -b "$BRANCH" "https://github.com/$REPO.git"
fi
cd caracal-1

echo "[1/6] install deps"
pip install -q -r requirements.txt

echo "[2/6] build datasets"
python -m data.ignite.build_omni_math --out "$OUT_ROOT/omni_math.jsonl"
python -m data.ignite.build_livecodebench --out "$OUT_ROOT/lcb.jsonl"
python -m data.ignite.build_bigcodebench --out "$OUT_ROOT/bcb.jsonl"
python -m data.ignite.build_matharena --out "$OUT_ROOT/matharena.jsonl"
python -m data.ignite.build_aime --out "$OUT_ROOT/aime.jsonl"
python -m data.ignite.build_putnam_lean --out "$OUT_ROOT/putnam.jsonl"

echo "[3/6] smoke tools"
python -m eval.ignite.tools.math_verify
python -m eval.ignite.tools.code_exec

echo "[4/6] Cond A baseline eval"
python -m train.ignite.A_baseline \
    --base "$BASE_MODEL" \
    --benches omni_math livecodebench matharena \
    --out "$OUT_ROOT/cond_A.json"

echo "[5/6] Cond C main outer loop (8 gens x 8 cands)"
python -m train.ignite.C_rsi_outer \
    --base "$BASE_MODEL" \
    --dataset-train "$OUT_ROOT/omni_math_train.jsonl" \
    --dataset-dev "$OUT_ROOT/omni_math_dev.jsonl" \
    --dataset-val "$OUT_ROOT/omni_math_val.jsonl" \
    --bench math --bench-name omni_math \
    --gens 8 --cands 8 --steps 150 \
    --out "$OUT_ROOT/cond_C_run" \
    --hf-repo iterate-labs-ai/ignite-3b-v1

echo "[6/6] analysis"
python -c "
from eval.ignite.stats import shapley_attribution, sigmoid_fit, bocpd
from pathlib import Path
import json
log = Path('$OUT_ROOT/cond_C_run/log.jsonl')
print('Shapley top-10:', shapley_attribution(log, 10))
gens = [json.loads(l) for l in open(log) if 'generation' in l]
vals = [g.get('val_r', 0) for g in gens if g.get('type') == 'generation']
if len(vals) >= 3:
    print('Sigmoid:', sigmoid_fit(list(range(1, len(vals)+1)), vals))
    print('BOCPD:', bocpd(vals))
"

echo "=== DONE. Model at https://huggingface.co/iterate-labs-ai/ignite-3b-v1 ==="

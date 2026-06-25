#!/usr/bin/env bash
# Pre-flight check antes de iniciar treino.
# Valida Kaggle CLI, Python deps, GPU.

set -e

echo "==> preflight check (Kaggle only)"

# 1. Kaggle CLI
echo "checking kaggle CLI..."
kaggle datasets list --max-size 1 > /dev/null 2>&1 \
    || (echo "ERROR: Kaggle CLI nao autenticado. ~/.kaggle/kaggle.json existe?" && exit 1)
echo "kaggle CLI ok"

# 2. Python deps
echo "checking python deps..."
python -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" || exit 1
python -c "import transformers; print(f'transformers {transformers.__version__}')" || exit 1
python -c "import unsloth" 2>/dev/null || echo "WARN: unsloth nao instalado (instala no notebook)"
python -c "import trl" 2>/dev/null || echo "WARN: trl nao instalado"
python -c "import datasets" 2>/dev/null || echo "WARN: datasets nao instalado"

# 3. GPU
echo "checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
else
    echo "WARN: sem nvidia-smi (so CPU?)"
fi

# 4. Disk
echo "checking disk..."
df -h . | tail -1

echo ""
echo "==> preflight OK · pronto pra treinar"

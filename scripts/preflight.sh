#!/usr/bin/env bash
# Pre-flight check antes de iniciar treino.
# Valida HF auth, W&B auth, GPU disponivel, requirements ok.

set -e

echo "==> preflight check"

# 1. HuggingFace auth
echo "checking huggingface-cli auth..."
huggingface-cli whoami || (echo "ERROR: HF auth missing. Run: huggingface-cli login" && exit 1)

# 2. W&B auth
echo "checking wandb auth..."
wandb status 2>/dev/null || (echo "WARN: wandb not authenticated. Run: wandb login")

# 3. Python deps
echo "checking python deps..."
python -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" || exit 1
python -c "import transformers; print(f'transformers {transformers.__version__}')" || exit 1
python -c "import unsloth" 2>/dev/null || echo "WARN: unsloth not installed"
python -c "import trl" || echo "WARN: trl not installed"

# 4. GPU availability
echo "checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
else
    echo "WARN: no nvidia-smi (CPU only?)"
fi

# 5. Disk space
echo "checking disk space..."
df -h . | tail -1

# 6. KERNEL-D integrity
echo "checking KERNEL-D integrity..."
if [ -f "harness/stop_pattern_detector.py" ]; then
    python harness/stop_pattern_detector.py | head -5
fi

echo ""
echo "==> preflight OK · ready to train"

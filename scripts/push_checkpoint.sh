#!/usr/bin/env bash
# Push checkpoint para HuggingFace Hub com revision tag.

set -e

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 <local_dir> <hf_repo> <revision_tag>"
    echo "Ex: $0 ./ckpt-out iterate-labs/caracal-base-pretrain step-5500"
    exit 1
fi

LOCAL_DIR=$1
HF_REPO=$2
REVISION=$3

echo "==> pushing $LOCAL_DIR to $HF_REPO@$REVISION"

huggingface-cli upload "$HF_REPO" "$LOCAL_DIR" . \
    --revision "$REVISION" \
    --commit-message "$REVISION by $(whoami) at $(date -Iseconds)"

echo "==> push complete · update SCHEDULE.md with revision $REVISION"

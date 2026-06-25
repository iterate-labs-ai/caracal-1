#!/usr/bin/env bash
# Publica checkpoint como Kaggle Dataset publico.
#
# Usage: bash scripts/publish_kaggle_dataset.sh <local_dir> <username> <dataset_slug>
# Exemplo: bash scripts/publish_kaggle_dataset.sh ./ckpt-out pamf2 caracal-base-step4000

set -e

LOCAL_DIR=$1
USERNAME=$2
SLUG=$3

if [ -z "$LOCAL_DIR" ] || [ -z "$USERNAME" ] || [ -z "$SLUG" ]; then
    echo "Usage: $0 <local_dir> <username> <dataset_slug>"
    exit 1
fi

# Criar metadata
cat > "$LOCAL_DIR/dataset-metadata.json" << EOF
{
  "title": "Caracal Base 3B $SLUG",
  "id": "$USERNAME/$SLUG",
  "licenses": [{"name": "apache-2.0"}]
}
EOF

echo "Publicando dataset $USERNAME/$SLUG..."
kaggle datasets create -p "$LOCAL_DIR" --public

echo ""
echo "Dataset disponivel em: kaggle.com/datasets/$USERNAME/$SLUG"
echo "Proximo no relay configura RESUME_DATASET = $USERNAME/$SLUG"

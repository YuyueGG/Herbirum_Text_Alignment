#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "${ROOT_DIR}/train.py" \
  --preset convnext44 \
  --train_jsonl examples/sample_jsonl/cyrtandra44_train.jsonl \
  --test_jsonl examples/sample_jsonl/cyrtandra44_test.jsonl \
  --label2id examples/sample_jsonl/label2id.json \
  --seed 42 \
  --save_dir outputs/convnext44_seed42 \
  --num_workers 8
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "${ROOT_DIR}/train.py" \
  --preset resnet9 \
  --train_jsonl examples/sample_jsonl/cyrtandra9_train.jsonl \
  --test_jsonl examples/sample_jsonl/cyrtandra9_test.jsonl \
  --label2id examples/sample_jsonl/label2id9.json \
  --seed 42 \
  --save_dir outputs/resnet9_seed42 \
  --num_workers 8
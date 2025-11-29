#!/bin/bash

echo "Running Jeopardy classifier..."
echo ""

python3 main.py \
  --input JEOPARDY_QUESTIONS1.json \
  --max_samples 1000 \
  --model_name Qwen/Qwen3-8B \
  --batch_size 32 \
  --n_samples 100

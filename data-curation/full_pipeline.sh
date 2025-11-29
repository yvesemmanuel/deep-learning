#!/bin/bash

echo "Running Jeopardy classifier..."
echo ""

python3 main.py \
  --input JEOPARDY_QUESTIONS1.json \
  --model_name Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --batch_size 64 \
  --n_samples 1000

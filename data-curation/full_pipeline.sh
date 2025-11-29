#!/bin/bash

echo "Running Jeopardy classifier..."
echo ""

python3 main.py \
  --input JEOPARDY_QUESTIONS1.json \
  --model_name Qwen/Qwen3-4B-Instruct-2507 \
  --batch_size 256 \
  --max_new_tokens 500 \
  --n_samples 1000 \
  --save_every_n 100

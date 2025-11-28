#!/bin/bash

echo "Running Jeopardy classifier..."
echo ""

python3 main.py \
  --input JEOPARDY_QUESTIONS1.json \
  --max_samples 10

# Jeopardy Dataset Curation for NER Validation

A pipeline for curating validation datasets from 216,930 Jeopardy questions to evaluate Named Entity Recognition (NER) algorithms across three challenging linguistic categories.

## Problem

NER algorithms need robust validation across diverse linguistic patterns. This pipeline curates stratified subsets from the Jeopardy corpus to enable systematic performance comparison across:

- **Phrases containing numbers** - Dates, measurements, years, quantities
- **Phrases containing non-English words** - Foreign terms, loanwords, transliterations
- **Phrases containing unusual proper nouns** - Obscure names, places, specialized terminology

## Dataset Overview

The source dataset (`JEOPARDY_QUESTIONS1.json`):
- **216,930** total questions
- **27,995** unique categories (e.g., "HISTORY", "SCIENCE", "LITERATURE")

## Results

Processing all the Jeopardy dataset, 216,930 questions:

| Category | Count | Percentage | Sample Size |
|----------|-------|------------|-------------|
| **Contains Numbers** | 80,275 | 37.01% | 1,000 |
| **Contains Non-English** | 37,168 | 17.13% | 1,000 |
| **Contains Unusual Proper Nouns** | 28,509 | 13.14% | 1,000 |

## How It Works

The pipeline has four components:

**1. Data Loading** - Loads and preprocesses the raw Jeopardy JSON dataset, handling text encoding and normalization.

**2. LLM Classification** - Uses a HuggingFace transformer (Qwen3-4B-Instruct) to classify each question with structured JSON output. The model analyzes question and answer text together, returning three boolean flags per question. Batch processing with GPU acceleration and periodic checkpointing enable efficient processing of the full dataset.

**3. Stratified Sampling** - Creates balanced 1,000-sample datasets by maintaining the original distribution of Jeopardy categories. This prevents over-representation of any single topic domain.

**4. Statistical Estimation** - Analyzes the classified dataset to calculate class prevalence across all 216,930 questions.

## How to Run

### Installation

```bash
pip install -r requirements.txt
```

**Requirements**: Python 3.8+, PyTorch, HuggingFace transformers, 8GB+ GPU memory

### Process Full Dataset

```bash
./full_pipeline.sh
```

Processes all 216,930 questions using Qwen3-4B-Instruct-2507 with batch size 256 and creates 1,000-sample curated datasets.

### Quick Test (1,000 samples)

```bash
./test_small_sample.sh
```

Processes 1,000 questions for testing using Qwen3-8B with batch size 32 and creates 100-sample curated datasets.

## Output

**Curated Datasets** (`output/`):
- `dataset_numbers.json` - 1,000 questions containing numbers
- `dataset_non_english.json` - 1,000 questions with non-English words
- `dataset_unusual_proper_nouns.json` - 1,000 questions with unusual proper nouns

**Statistics**: `output/classification_stats.json` - Complete classification results and prevalence estimates

Each entry includes original Jeopardy fields plus three classification flags (`llm_has_numbers`, `llm_has_non_english`, `llm_has_unusual_proper_nouns`).

## Architecture

```
data-curation/
├── src/
│   ├── classifier.py        # LLM-based classification
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── curator.py           # Stratified sampling and estimation
│   └── logger.py            # Logging utilities
└── main.py                  # Pipeline orchestration
```

## Performance

- **Processing time**: ~2-4 hours for full dataset (4B model on GPU)
- **GPU acceleration**: Strongly recommended
- **Checkpointing**: Automatic saves enable resumption

**Optimization**:
- Increase `--batch_size` for faster processing (if GPU memory allows)
- Use smaller models for memory-constrained environments
- Process in chunks with `--max_samples` for testing

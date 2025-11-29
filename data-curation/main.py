"""
Main script for classifying Jeopardy questions.
"""

import argparse
from src.data_loader import JeopardyDataLoader
from src.classifier import HuggingFaceJeopardyClassifier
from src.curator import DatasetCurator
from src.logger import logger


def main():
    """Main execution flow."""
    parser = argparse.ArgumentParser(
        description="Classify Jeopardy questions using HuggingFace with custom JSON parsing"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="JEOPARDY_QUESTIONS1.json",
        help="Input JSON file path",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="Language model to use from HF (default: Qwen/Qwen3-30B-A3B-Instruct-2507)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Inference batch size (default: 32)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save curated datasets (default: output)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="Number of samples per curated dataset (default: 1000)",
    )
    parser.add_argument(
        "--predictions_file",
        type=str,
        default="predictions.json",
        help="Path to save/load predictions (default: predictions.json)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("JEOPARDY QUESTION CLASSIFIER")
    logger.info("Using HuggingFace with custom JSON parsing for structured outputs")
    logger.info("=" * 60)

    logger.info("\n[1/4] Loading and preprocessing data...")
    loader = JeopardyDataLoader(args.input)
    data = loader.load_and_clean(max_samples=args.max_samples)

    stats = loader.get_statistics(data)
    logger.info("\nDataset statistics:")
    logger.info(f"  Total questions: {stats['total_questions']:,}")
    logger.info(f"  Unique categories: {stats['unique_categories']:,}")
    logger.info(f"  Unique shows: {stats['unique_shows']:,}")

    total_dataset_size = len(data)
    logger.info(f"  Full dataset size: {total_dataset_size:,}")

    logger.info("\n[2/4] Classifying questions...")
    classifier = HuggingFaceJeopardyClassifier(
        model_name=args.model_name, batch_size=args.batch_size
    )
    predictions = classifier.classify_dataset(data, 1, args.predictions_file)

    logger.info("\n[3/4] Curating datasets...")
    curator = DatasetCurator(output_dir=args.output_dir)
    _ = curator.create_curated_datasets(
        predictions, n_samples=args.n_samples, total_dataset_size=total_dataset_size
    )

    logger.info("\n[4/4] Pipeline complete!")
    logger.info("=" * 60)
    logger.info("\nSummary:")
    logger.info(f"  Input: {args.input}")
    logger.info(f"  Classified: {len(predictions):,} questions")
    logger.info(f"  Curated datasets saved to: {args.output_dir}/")
    logger.info(f"  Statistics saved to: {args.output_dir}/classification_stats.json")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

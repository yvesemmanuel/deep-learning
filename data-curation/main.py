"""
Main script for classifying Jeopardy questions.
"""

import argparse
from src.data_loader import JeopardyDataLoader
from src.classifier import HuggingFaceJeopardyClassifier
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

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("JEOPARDY QUESTION CLASSIFIER")
    logger.info("Using HuggingFace with custom JSON parsing for structured outputs")
    logger.info("=" * 60)

    logger.info("\n[1/3] Loading and preprocessing data...")
    loader = JeopardyDataLoader(args.input)
    data = loader.load_and_clean(max_samples=args.max_samples)

    stats = loader.get_statistics(data)
    logger.info("\nDataset statistics:")
    logger.info(f"  Total questions: {stats['total_questions']:,}")
    logger.info(f"  Unique categories: {stats['unique_categories']:,}")
    logger.info(f"  Unique shows: {stats['unique_shows']:,}")

    classifier = HuggingFaceJeopardyClassifier()

    predictions = classifier.classify_dataset(data, 1, "predictions.json")


if __name__ == "__main__":
    main()

"""
Dataset curation utilities for creating filtered datasets.
"""

import json
from typing import List, Dict, Optional
from pathlib import Path
from collections import Counter
from .logger import logger


class DatasetCurator:
    """Creates curated datasets from classified data."""

    def __init__(self, output_dir: str = "output"):
        """
        Initialize the curator.

        Parameters:
        -----------
        output_dir: Directory to save curated datasets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_predictions(self, predictions_file: str) -> List[Dict]:
        """
        Load predictions from JSON file.

        Parameters:
        -----------
        predictions_file: Path to predictions JSON file

        Returns:
        --------
        List of predictions
        """
        logger.info(f"Loading predictions from {predictions_file}")
        with open(predictions_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data):,} predictions")
        return data

    def analyze_classification_quality(self, data: List[Dict]) -> Dict:
        """
        Analyze the quality of classifications, identifying failed predictions.

        Parameters:
        -----------
        data: Classified data

        Returns:
        --------
        Dictionary with quality statistics
        """
        total = len(data)
        failed = 0
        successful = 0

        for row in data:
            has_numbers = row.get("llm_has_numbers")
            has_non_english = row.get("llm_has_non_english")
            has_unusual = row.get("llm_has_unusual_proper_nouns")

            if has_numbers is None and has_non_english is None and has_unusual is None:
                failed += 1
            else:
                successful += 1

        logger.info("\nClassification Quality Analysis:")
        logger.info(f"  Total predictions: {total:,}")
        logger.info(f"  Successful: {successful:,} ({100 * successful / total:.2f}%)")
        logger.info(
            f"  Failed (all fields None): {failed:,} ({100 * failed / total:.2f}%)"
        )

        return {
            "total": total,
            "successful": successful,
            "failed": failed,
            "success_rate": round(100 * successful / total, 2) if total > 0 else 0,
            "failure_rate": round(100 * failed / total, 2) if total > 0 else 0,
        }

    def filter_by_classification(
        self,
        data: List[Dict],
        has_numbers: Optional[bool] = None,
        has_non_english: Optional[bool] = None,
        has_unusual_proper_nouns: Optional[bool] = None,
    ) -> List[Dict]:
        """
        Filter data by classification criteria.

        Parameters:
        -----------
        data: Classified data
        has_numbers: Filter by has_numbers (None to ignore)
        has_non_english: Filter by has_non_english (None to ignore)
        has_unusual_proper_nouns: Filter by has_unusual_proper_nouns (None to ignore)

        Returns:
        --------
        Filtered data
        """
        filtered = data

        if has_numbers is not None:
            filtered = [
                row for row in filtered if row.get("llm_has_numbers") == has_numbers
            ]

        if has_non_english is not None:
            filtered = [
                row
                for row in filtered
                if row.get("llm_has_non_english") == has_non_english
            ]

        if has_unusual_proper_nouns is not None:
            filtered = [
                row
                for row in filtered
                if row.get("llm_has_unusual_proper_nouns") == has_unusual_proper_nouns
            ]

        return filtered

    def stratified_sample(
        self, data: List[Dict], n_samples: int, stratify_by: str = "category"
    ) -> List[Dict]:
        """
        Create a stratified sample from the data.

        Parameters:
        -----------
        data: Data to sample from
        n_samples: Target number of samples
        stratify_by: Field to stratify by (default: 'category')

        Returns:
        --------
        Stratified sample
        """
        if len(data) <= n_samples:
            return data

        strata_counts = Counter(row.get(stratify_by, "unknown") for row in data)

        samples_per_stratum = {}
        for stratum, count in strata_counts.items():
            samples_per_stratum[stratum] = max(1, int(n_samples * count / len(data)))

        sampled = []
        for stratum in strata_counts:
            stratum_data = [
                row for row in data if row.get(stratify_by, "unknown") == stratum
            ]
            n = min(samples_per_stratum[stratum], len(stratum_data))
            sampled.extend(stratum_data[:n])  # Take first n (could randomize here)

        if len(sampled) > n_samples:
            sampled = sampled[:n_samples]
        elif len(sampled) < n_samples:
            # Add more from largest strata
            remaining_data = [row for row in data if row not in sampled]
            sampled.extend(remaining_data[: n_samples - len(sampled)])

        return sampled

    def create_curated_datasets(
        self,
        data: List[Dict],
        n_samples: int = 1000,
        total_dataset_size: Optional[int] = None,
    ) -> dict:
        """
        Create the 3 curated datasets from classified data.

        Parameters:
        -----------
        data: Classified data
        n_samples: Number of samples per curated dataset
        total_dataset_size: Total size of the full dataset for estimation (e.g., 216930)

        Returns:
        --------
        Dictionary with curated datasets
        """
        logger.info(f"\nCreating curated datasets with {n_samples} samples each...")

        quality_stats = self.analyze_classification_quality(data)

        valid_data = [
            row
            for row in data
            if not (
                row.get("llm_has_numbers") is None
                and row.get("llm_has_non_english") is None
                and row.get("llm_has_unusual_proper_nouns") is None
            )
        ]

        logger.info(f"\nUsing {len(valid_data):,} valid predictions for curation")

        if len(valid_data) == 0:
            logger.error("\nERROR: No valid predictions to curate!")
            logger.error("All classifications failed. This likely means:")
            logger.error("  1. The model is not generating valid JSON responses")
            logger.error("  2. The JSON parsing is failing")
            logger.error("  3. Check the predictions file to see raw model outputs")

            return {
                "numbers": [],
                "non_english": [],
                "unusual_proper_nouns": [],
                "stats": {
                    "classification_quality": quality_stats,
                    "classified_dataset": {
                        "total_questions": len(data),
                        "valid_predictions": 0,
                    },
                    "error": "No valid predictions - all classifications failed",
                },
            }

        data_numbers = self.filter_by_classification(valid_data, has_numbers=True)
        data_non_english = self.filter_by_classification(
            valid_data, has_non_english=True
        )
        data_unusual = self.filter_by_classification(
            valid_data, has_unusual_proper_nouns=True
        )

        logger.info("\nClass distributions in classified dataset:")
        logger.info(
            f"  Numbers: {len(data_numbers):,} ({100 * len(data_numbers) / len(valid_data):.2f}%)"
        )
        logger.info(
            f"  Non-English: {len(data_non_english):,} ({100 * len(data_non_english) / len(valid_data):.2f}%)"
        )
        logger.info(
            f"  Unusual Proper Nouns: {len(data_unusual):,} ({100 * len(data_unusual) / len(valid_data):.2f}%)"
        )

        if total_dataset_size and total_dataset_size > len(valid_data):
            logger.info(
                f"\nEstimated counts in full dataset ({total_dataset_size:,} questions):"
            )
            logger.info(
                f"  Numbers: ~{int(len(data_numbers) / len(valid_data) * total_dataset_size):,} "
                f"({100 * len(data_numbers) / len(valid_data):.2f}%)"
            )
            logger.info(
                f"  Non-English: ~{int(len(data_non_english) / len(valid_data) * total_dataset_size):,} "
                f"({100 * len(data_non_english) / len(valid_data):.2f}%)"
            )
            logger.info(
                f"  Unusual Proper Nouns: ~{int(len(data_unusual) / len(valid_data) * total_dataset_size):,} "
                f"({100 * len(data_unusual) / len(valid_data):.2f}%)"
            )

        sample_numbers = self.stratified_sample(data_numbers, n_samples)
        sample_non_english = self.stratified_sample(data_non_english, n_samples)
        sample_unusual = self.stratified_sample(data_unusual, n_samples)

        self._save_dataset(sample_numbers, "dataset_numbers.json")
        self._save_dataset(sample_non_english, "dataset_non_english.json")
        self._save_dataset(sample_unusual, "dataset_unusual_proper_nouns.json")

        logger.info(f"\nCurated datasets saved to {self.output_dir}:")
        logger.info(f"  dataset_numbers.json: {len(sample_numbers)} samples")
        logger.info(f"  dataset_non_english.json: {len(sample_non_english)} samples")
        logger.info(
            f"  dataset_unusual_proper_nouns.json: {len(sample_unusual)} samples"
        )

        stats = {
            "classification_quality": quality_stats,
            "classified_dataset": {
                "total_questions": len(data),
                "valid_predictions": len(valid_data),
            },
            "numbers": {
                "classified_count": len(data_numbers),
                "percentage_of_valid": round(
                    100 * len(data_numbers) / len(valid_data), 2
                ),
                "sample_size": len(sample_numbers),
            },
            "non_english": {
                "classified_count": len(data_non_english),
                "percentage_of_valid": round(
                    100 * len(data_non_english) / len(valid_data), 2
                ),
                "sample_size": len(sample_non_english),
            },
            "unusual_proper_nouns": {
                "classified_count": len(data_unusual),
                "percentage_of_valid": round(
                    100 * len(data_unusual) / len(valid_data), 2
                ),
                "sample_size": len(sample_unusual),
            },
        }

        if total_dataset_size and total_dataset_size > len(valid_data):
            stats["full_dataset_estimates"] = {
                "total_size": total_dataset_size,
                "numbers": {
                    "estimated_count": int(
                        len(data_numbers) / len(valid_data) * total_dataset_size
                    ),
                    "percentage": round(100 * len(data_numbers) / len(valid_data), 2),
                },
                "non_english": {
                    "estimated_count": int(
                        len(data_non_english) / len(valid_data) * total_dataset_size
                    ),
                    "percentage": round(
                        100 * len(data_non_english) / len(valid_data), 2
                    ),
                },
                "unusual_proper_nouns": {
                    "estimated_count": int(
                        len(data_unusual) / len(valid_data) * total_dataset_size
                    ),
                    "percentage": round(100 * len(data_unusual) / len(valid_data), 2),
                },
            }

        with open(self.output_dir / "classification_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(
            f"\nStatistics saved to {self.output_dir / 'classification_stats.json'}"
        )

        return {
            "numbers": sample_numbers,
            "non_english": sample_non_english,
            "unusual_proper_nouns": sample_unusual,
            "stats": stats,
        }

    def _save_dataset(self, data: List[Dict], filename: str):
        """Save a dataset to file."""
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

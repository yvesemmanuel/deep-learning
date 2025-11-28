"""
Data loading and preprocessing utilities.
"""

import json
from typing import List, Dict, Optional
from pathlib import Path
from .logger import logger


class JeopardyDataLoader:
    """Handles loading and preprocessing of Jeopardy dataset."""

    def __init__(self, data_path: str):
        """
        Initialize the data loader.

        Parameters:
        -----------
        data_path: Path to the JSON data file
        """
        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

    def load_data(self, max_samples: Optional[int] = None) -> List[Dict]:
        """
        Load data from JSON file.

        Parameters:
        -----------
        max_samples: Maximum number of samples to load (None for all)

        Returns:
        --------
        List of question dictionaries
        """
        logger.info(f"Loading data from {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if max_samples is not None:
            data = data[:max_samples]
            logger.info(f"Loaded {len(data):,} samples (limited to {max_samples:,})")
        else:
            logger.info(f"Loaded {len(data):,} samples")

        return data

    def clean_data(self, data: List[Dict]) -> List[Dict]:
        """
        Clean and preprocess the data.

        Parameters:
        -----------
        data: List of question dictionaries

        Returns:
        --------
        Cleaned data
        """
        logger.info("Cleaning data...")

        for row in data:
            row["category"] = str(row.get("category", "")).strip()
            row["question"] = str(row.get("question", "")).strip()
            row["answer"] = str(row.get("answer", "")).strip()
            row["air_date"] = str(row.get("air_date", ""))
            row["value"] = row.get("value")
            row["round"] = str(row.get("round", ""))
            row["show_number"] = str(row.get("show_number", ""))

            row.pop("llm_has_numbers", None)
            row.pop("llm_has_non_english", None)
            row.pop("llm_has_unusual_proper_nouns", None)

        logger.info(f"Cleaned {len(data):,} records")
        return data

    def load_and_clean(self, max_samples: Optional[int] = None) -> List[Dict]:
        """
        Load and clean data in one step.

        Parameters:
        -----------
        max_samples: Maximum number of samples to load (None for all)

        Returns:
        --------
        Cleaned data ready for classification
        """
        data = self.load_data(max_samples=max_samples)
        data = self.clean_data(data)
        return data

    def get_statistics(self, data: List[Dict]) -> Dict[str, int]:
        """
        Get basic statistics about the dataset.

        Parameters:
        -----------
        data: Data to analyze

        Returns:
        --------
        Dictionary with statistics
        """
        stats = {
            "total_questions": len(data),
            "unique_categories": len(set(row["category"] for row in data)),
            "unique_shows": len(set(row.get("show_number", "") for row in data)),
        }

        if any("llm_has_numbers" in row for row in data):
            stats["has_numbers"] = sum(1 for row in data if row.get("llm_has_numbers"))
            stats["has_non_english"] = sum(
                1 for row in data if row.get("llm_has_non_english")
            )
            stats["has_unusual_proper_nouns"] = sum(
                1 for row in data if row.get("llm_has_unusual_proper_nouns")
            )

        return stats

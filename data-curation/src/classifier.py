"""
HuggingFace-based classifier for Jeopardy questions.
Uses custom JSON parsing for structured outputs with batch processing.
"""

import json
from typing import List, Dict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from .models import JeopardyClassification
from .parser import parse_json_response
from .logger import logger


class HuggingFaceJeopardyClassifier:
    """
    LLM-based classifier for Jeopardy questions using HuggingFace.
    Uses batch processing for efficient classification with custom JSON parsing.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-1.7B",
        batch_size: int = 8,
        device_map: str = "auto",
        max_new_tokens: int = 500,
    ):
        """
        Initialize the HuggingFace-based classifier.

        Parameters:
        -----------
        model_name: HuggingFace model to use (default: Qwen3-1.7B)
        batch_size: Number of items to process in each batch
        device_map: Device mapping strategy for model loading
        max_new_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

        logger.info(f"Loading model: {model_name}")
        logger.info("This may take a few moments...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        self.pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
        )

        logger.info(f"Model loaded successfully on device: {device_map}")

    def _create_prompt(self, category: str, question: str, answer: str) -> str:
        """Create the classification prompt for a single example."""

        prompt = f"""Analyze this Jeopardy clue and classify it according to three criteria.

Category: {category}
Question: {question}
Answer: {answer}

Classify as true or false for each criterion:

1. has_numbers: Does the text contain any numbers? This includes:
   - Digits (1, 2, 3, 1492, 1.5)
   - Written numbers (one, two, hundred, million)
   - Years, ages
   - Monetary amounts ($100, 5 dollars)
   - Measurements, statistics, quantities

2. has_non_english: Does the text contain non-English words or phrases? This includes:
   - Foreign words/phrases (Latin, French, Spanish, German, Italian, etc.)
   - Loanwords that are clearly foreign (not fully anglicized)
   - Non-English names of places, works, or concepts
   - Words with diacritics or non-Latin characters
   - Scientific Latin names (genus/species)

3. has_unusual_proper_nouns: Does the text contain unusual or obscure proper nouns? This includes:
   - Obscure historical figures (not widely known)
   - Uncommon place names (small towns, obscure geographic features)
   - Specialized terminology names
   - Mythological/religious figures (except very common ones like Jesus, Zeus)
   - Foreign names that would be unfamiliar to average Americans
   - Obscure literary/artistic figures

You must respond with ONLY a valid JSON object in this exact format:
{{
  "has_numbers": true/false,
  "has_non_english": true/false,
  "has_unusual_proper_nouns": true/false
}}

Do not include any explanation or additional text. Only return the JSON object."""

        return prompt

    def classify_batch(self, rows: List[Dict]) -> List[JeopardyClassification]:
        """
        Classify a batch of rows using structured outputs.

        Parameters:
        -----------
        rows: List of dictionaries with 'category', 'question', and 'answer' keys

        Returns:
        --------
        List of JeopardyClassification objects
        """
        classifications = []

        for row in rows:
            prompt = self._create_prompt(
                category=row.get("category", ""),
                question=row.get("question", ""),
                answer=row.get("answer", ""),
            )

            try:
                response = self.pipeline(prompt)

                generated_text = response[0]["generated_text"]

                classification = parse_json_response(generated_text)
                classifications.append(classification)

            except Exception as e:
                logger.warning(f"Error classifying row: {e}")
                classifications.append(
                    JeopardyClassification(
                        has_numbers=False,
                        has_non_english=False,
                        has_unusual_proper_nouns=False,
                    )
                )

        return classifications

    def classify_dataset(
        self,
        data: List[Dict],
        save_every: int = 100,
        output_path: str = "classified_results.json",
    ) -> List[Dict]:
        """
        Classify entire dataset with batch processing and checkpointing.

        Parameters:
        -----------
        data: List of question dictionaries
        save_every: Save checkpoint every N questions
        output_path: Path to save results

        Returns:
        --------
        List of dictionaries with classification results added
        """
        total_rows = len(data)
        logger.info(
            f"\nClassifying {total_rows:,} questions with model {self.model_name}"
        )
        logger.info(f"Batch size: {self.batch_size}")

        num_batches = (total_rows + self.batch_size - 1) // self.batch_size

        with tqdm(total=total_rows, desc="Processing questions") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, total_rows)
                batch = data[start_idx:end_idx]

                classifications = self.classify_batch(batch)

                for row, classification in zip(batch, classifications):
                    row["llm_has_numbers"] = classification.has_numbers
                    row["llm_has_non_english"] = classification.has_non_english
                    row["llm_has_unusual_proper_nouns"] = (
                        classification.has_unusual_proper_nouns
                    )

                pbar.update(len(batch))

                if (end_idx) % save_every == 0 or end_idx == total_rows:
                    with open(output_path, "w") as f:
                        json.dump(data, f, indent=2)
                    logger.info(
                        f"\nCheckpoint saved at question {end_idx}/{total_rows}"
                    )

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"\nClassification complete! Results saved to {output_path}")

        return data

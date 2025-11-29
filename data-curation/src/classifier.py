"""
HuggingFace-based classifier for Jeopardy questions.
Uses custom JSON parsing for structured outputs with batch processing.
"""

import json
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        tokenizer.padding_side = "left"

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device_map, dtype="auto"
        )

        self.pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            batch_size=batch_size,
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

    def _parse_responses_parallel(
        self, responses: List[List[Dict]], max_workers: int = 8
    ) -> List[JeopardyClassification]:
        """
        Parse responses in parallel using threads for faster processing.

        Parameters:
        -----------
        responses: List of response objects from the pipeline
        max_workers: Maximum number of threads to use

        Returns:
        --------
        List of JeopardyClassification objects
        """

        def parse_single_response(response) -> JeopardyClassification:
            """Parse a single response and return classification."""
            try:
                generated_text = response[0]["generated_text"]
                return parse_json_response(generated_text)
            except Exception as e:
                # Log first few failures with full details for debugging
                if not hasattr(parse_single_response, "error_count"):
                    parse_single_response.error_count = 0

                parse_single_response.error_count += 1

                if parse_single_response.error_count <= 3:
                    logger.warning(
                        f"Error parsing response #{parse_single_response.error_count}: {e}"
                    )
                    try:
                        logger.warning(f"Response structure: {response}")
                        if response and len(response) > 0:
                            logger.warning(
                                f"Generated text preview: {str(response[0].get('generated_text', 'N/A'))[:200]}"
                            )
                    except:
                        pass

                return JeopardyClassification(
                    has_numbers=None,
                    has_non_english=None,
                    has_unusual_proper_nouns=None,
                )

        classifications: List[JeopardyClassification] = [
            JeopardyClassification(
                has_numbers=None,
                has_non_english=None,
                has_unusual_proper_nouns=None,
            )
        ] * len(responses)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(parse_single_response, response): idx
                for idx, response in enumerate(responses)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                classifications[idx] = future.result()

        return classifications

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
        # Create all prompts for batch processing
        prompts = [
            self._create_prompt(
                category=row.get("category", ""),
                question=row.get("question", ""),
                answer=row.get("answer", ""),
            )
            for row in rows
        ]

        try:
            responses = self.pipeline(prompts)

            classifications = self._parse_responses_parallel(responses)

        except Exception as e:
            logger.warning(f"Error processing batch: {e}")
            classifications = [
                JeopardyClassification(
                    has_numbers=None,
                    has_non_english=None,
                    has_unusual_proper_nouns=None,
                )
                for _ in rows
            ]

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

"""
Pydantic models for structured classification outputs.
"""

from pydantic import BaseModel, Field


class JeopardyClassification(BaseModel):
    """
    Structured output for Jeopardy question classification.
    """

    has_numbers: bool = Field(
        description="Does the text contain any numbers? Includes digits, written numbers, years, dates, ages, monetary amounts, ordinals, measurements, or statistics."
    )
    has_non_english: bool = Field(
        description="Does the text contain non-English words or phrases? Includes foreign words, loanwords, non-English names, words with diacritics, or scientific Latin names."
    )
    has_unusual_proper_nouns: bool = Field(
        description="Does the text contain unusual or obscure proper nouns? Includes obscure historical figures, uncommon place names, specialized terminology, mythological/religious figures, foreign names, or obscure literary/artistic figures."
    )

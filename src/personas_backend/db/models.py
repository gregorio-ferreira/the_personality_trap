"""
Database models for the personality trap research project.

This module contains the main data models based on the production
materialized view population.exp20250312_populations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel

from .schema_config import get_table_args


class Persona(SQLModel, table=True):  # type: ignore[call-arg]
    """
    Persona model based on population.exp20250312_populations view.

    Contains all demographic and descriptive information for personas
    used in the personality research experiments.
    """

    __tablename__ = "personas"
    __table_args__ = get_table_args()

    # Primary key (auto-increment)
    id: int | None = Field(
        default=None,
        primary_key=True,
        sa_column_kwargs={"autoincrement": True},
        description="Auto-increment primary key",
    )

    # Reference fields from original data
    ref_personality_id: Optional[int] = Field(
        default=None, index=True, description="Reference to original personality ID"
    )
    population: Optional[str] = Field(
        default=None,
        index=True,
        max_length=100,
        description="Population source (e.g., spain826, borderline_maxN_gpt4o)",
    )
    model: Optional[str] = Field(
        default=None,
        index=True,
        max_length=100,
        description="LLM model used to generate persona",
    )

    # Core persona information
    name: Optional[str] = Field(default=None, max_length=255, description="Persona name")
    age: Optional[int] = Field(default=None, description="Persona age")

    # Original and cleaned demographic fields
    gender_original: Optional[str] = Field(
        default=None, max_length=50, description="Original gender as generated"
    )
    gender: Optional[str] = Field(
        default=None,
        max_length=20,
        index=True,
        description="Cleaned gender (female, male, non-binary, other)",
    )

    race_original: Optional[str] = Field(
        default=None, max_length=100, description="Original race as generated"
    )
    race: Optional[str] = Field(
        default=None,
        max_length=50,
        index=True,
        description="Cleaned race (White, Black, Asian, Hispanic, Other)",
    )

    sexual_orientation_original: Optional[str] = Field(
        default=None, max_length=100, description="Original sexual orientation"
    )
    sexual_orientation: Optional[str] = Field(
        default=None,
        max_length=20,
        index=True,
        description="Cleaned sexual orientation (heterosexual, lgbtq+, etc.)",
    )

    ethnicity: Optional[str] = Field(default=None, max_length=100, description="Ethnicity")

    religious_belief_original: Optional[str] = Field(
        default=None, max_length=100, description="Original religious belief"
    )
    religious_belief: Optional[str] = Field(
        default=None,
        max_length=50,
        index=True,
        description="Cleaned religious belief (Christian, Islam, etc.)",
    )

    occupation_original: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Original occupation as generated",
    )
    occupation: Optional[str] = Field(
        default=None,
        max_length=100,
        index=True,
        description="Cleaned occupation category",
    )

    political_orientation_original: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Original political orientation",
    )
    political_orientation: Optional[str] = Field(
        default=None,
        max_length=20,
        index=True,
        description="Cleaned political orientation (Progressive, Centre)",
    )

    location_original: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Original location as generated",
    )
    location: Optional[str] = Field(
        default=None,
        max_length=100,
        index=True,
        description="Cleaned location (major cities or Other)",
    )

    # Description and metadata
    description: Optional[str] = Field(
        default=None, description="Full persona description/narrative"
    )
    word_count_description: Optional[int] = Field(
        default=None, description="Word count of description"
    )
    repetitions: Optional[int] = Field(default=1, description="Number of repetitions/generations")

    # Paper nomenclature columns
    model_print: Optional[str] = Field(
        default=None,
        max_length=50,
        index=True,
        description="Model name for paper display (e.g., GPT-4o, Claude-3.5-s)",
    )
    population_print: Optional[str] = Field(
        default=None,
        max_length=50,
        index=True,
        description="Population name for paper display (Base, Max N, Max P)",
    )


class ReferenceQuestionnaire(SQLModel, table=True):  # type: ignore[call-arg]
    """
    Reference questionnaire responses used as input for persona generation.

    Contains the Big Five personality questionnaire responses that were used
    as the basis for generating the personas in the research experiments.
    """

    __tablename__ = "reference_questionnaires"
    __table_args__ = get_table_args()

    # Primary key (auto-increment)
    id: int | None = Field(
        default=None,
        primary_key=True,
        sa_column_kwargs={"autoincrement": True},
        description="Auto-increment primary key",
    )

    # Reference to personality ID
    personality_id: Optional[int] = Field(
        default=None, index=True, description="Reference personality ID (1-826)"
    )

    # Question details
    question_number: Optional[int] = Field(
        default=None, index=True, description="Question number (1-24)"
    )
    question: Optional[str] = Field(default=None, description="Full text of the question")

    # Question categorization
    category: Optional[str] = Field(
        default=None,
        max_length=10,
        index=True,
        description="Big Five category (N=Neuroticism, E=Extraversion, etc.)",
    )
    key: Optional[bool] = Field(
        default=None, description="Whether this is a key question for the category"
    )

    # Response
    answer: Optional[bool] = Field(default=None, description="True/False answer to the question")


class RandomQuestionnaire(SQLModel, table=True):  # type: ignore[call-arg]
    """
    Random questionnaire responses (data migration from reference_questionnaires).

    Contains the Big Five personality questionnaire responses that were used
    as the basis for generating the personas in the research experiments.
    This table has the same structure as ReferenceQuestionnaire.
    """

    __tablename__ = "random_questionnaires"
    __table_args__ = get_table_args()

    # Primary key (auto-increment)
    id: int | None = Field(
        default=None,
        primary_key=True,
        sa_column_kwargs={"autoincrement": True},
        description="Auto-increment primary key",
    )

    # Reference to personality ID
    personality_id: Optional[int] = Field(
        default=None, index=True, description="Reference personality ID (1-826)"
    )

    # Question details
    question_number: Optional[int] = Field(
        default=None, index=True, description="Question number (1-24)"
    )
    question: Optional[str] = Field(default=None, description="Full text of the question")

    # Question categorization
    category: Optional[str] = Field(
        default=None,
        max_length=10,
        index=True,
        description="Big Five category (N=Neuroticism, E=Extraversion, etc.)",
    )
    key: Optional[bool] = Field(
        default=None, description="Whether this is a key question for the category"
    )

    # Response
    answer: Optional[bool] = Field(default=None, description="True/False answer to the question")


class ExperimentsGroup(SQLModel, table=True):  # type: ignore[call-arg]
    """
    Experiment groups that organize related experiments together.

    Contains configuration and metadata for groups of related experiments
    in the personality research project.
    """

    __tablename__ = "experiments_groups"
    __table_args__ = get_table_args()

    # Primary key (auto-increment)
    experiments_group_id: int | None = Field(
        default=None,
        primary_key=True,
        sa_column_kwargs={"autoincrement": True},
        description="Auto-increment primary key",
    )

    # Timestamps (server-side default for new rows, allows explicit values on restore)
    created: Optional[str] = Field(
        default=None,
        sa_column_kwargs={"server_default": "CURRENT_TIMESTAMP"},
        description="Creation timestamp (auto-generated or explicit)",
    )

    # Configuration
    description: Optional[str] = Field(
        default=None, description="Description of the experiment group"
    )
    system_role: Optional[str] = Field(default=None, description="System role prompt")
    base_prompt: Optional[str] = Field(default=None, description="Base prompt template")

    # Status flags
    concluded: Optional[bool] = Field(
        default=False, description="Whether the experiment group is concluded"
    )
    processed: Optional[bool] = Field(
        default=False, description="Whether the results have been processed"
    )
    translated: Optional[bool] = Field(
        default=False, description="Whether translations have been applied"
    )

    # LLM parameters
    temperature: Optional[float] = Field(default=None, description="Temperature parameter for LLM")
    top_p: Optional[float] = Field(default=1.0, description="Top-p parameter for LLM")
    ii_repeated: Optional[bool] = Field(
        default=None, description="Whether instructions are repeated"
    )


class ExperimentsList(SQLModel, table=True):  # type: ignore[call-arg]
    """
    Individual experiments within experiment groups.

    Contains the actual experiment execution records with parameters
    and results for each personality/model combination.
    """

    __tablename__ = "experiments_list"
    __table_args__ = get_table_args()

    # Primary key (auto-increment)
    experiment_id: int | None = Field(
        default=None,
        primary_key=True,
        sa_column_kwargs={"autoincrement": True},
        description="Auto-increment primary key",
    )

    # Foreign key to experiments group
    experiments_group_id: Optional[int] = Field(
        default=None, index=True, description="Reference to experiment group"
    )

    # Experiment configuration
    repeated: Optional[int] = Field(default=None, description="Repetition number")

    # Timestamp (server-side default for new rows, allows explicit values on restore)
    created: Optional[str] = Field(
        default=None,
        sa_column_kwargs={"server_default": "CURRENT_TIMESTAMP"},
        description="Creation timestamp (auto-generated or explicit)",
    )

    # Questionnaire setup
    questionnaire: Optional[str] = Field(
        default=None, description="Type of questionnaire (epqra, bigfive)"
    )
    language_instructions: Optional[str] = Field(
        default=None, description="Language for instructions"
    )
    language_questionnaire: Optional[str] = Field(
        default=None, description="Language for questionnaire"
    )

    # Model configuration
    model_provider: Optional[str] = Field(
        default=None, description="LLM provider (openai, bedrock, etc.)"
    )
    model: Optional[str] = Field(default=None, description="Specific model name")

    # Population and personality
    population: Optional[str] = Field(default=None, description="Population source")
    personality_id: Optional[int] = Field(
        default=None, index=True, description="Reference to personality"
    )

    # Execution details
    succeeded: Optional[bool] = Field(default=None, description="Whether experiment succeeded")
    llm_explanation: Optional[str] = Field(
        default=None, description="LLM explanation from answered questionnaire"
    )

    # Metadata
    repo_sha: Optional[str] = Field(default=None, description="Git commit SHA")


class ExperimentRequestMetadata(SQLModel, table=True):  # type: ignore[call-arg]
    """Request/response payloads captured during experiment execution."""

    __tablename__ = "experiment_request_metadata"
    __table_args__ = get_table_args()

    # Primary key (auto-increment)
    id: int | None = Field(
        default=None,
        primary_key=True,
        sa_column_kwargs={"autoincrement": True},
        description="Auto-increment primary key",
    )

    # Timestamp (server-side default)
    created: datetime | None = Field(
        default=None,
        sa_column_kwargs={"server_default": "CURRENT_TIMESTAMP"},
        description="Creation timestamp (auto-generated)",
    )

    experiment_id: Optional[int] = Field(default=None, index=True)
    request_json: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSONB))
    response_json: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSONB))
    request_metadata: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSONB))


class EvalQuestionnaires(SQLModel, table=True):  # type: ignore[call-arg]
    """
    Questionnaire evaluation results.

    Contains the parsed answers from LLMs for each question
    in the personality questionnaires.
    """

    __tablename__ = "eval_questionnaires"
    __table_args__ = get_table_args()

    # Primary key (auto-increment)
    id: int | None = Field(
        default=None,
        primary_key=True,
        sa_column_kwargs={"autoincrement": True},
        description="Auto-increment primary key",
    )

    # Foreign key to experiment
    experiment_id: Optional[int] = Field(
        default=None, index=True, description="Reference to experiment"
    )

    # Question details
    question_number: Optional[int] = Field(
        default=None, index=True, description="Question number in questionnaire"
    )

    # Answer
    answer: Optional[int] = Field(default=None, description="Parsed answer value")


class Questionnaire(SQLModel, table=True):  # type: ignore[call-arg]
    """
    Questionnaire definitions and questions.

    Contains the actual questions, categories, and metadata for
    different personality questionnaires (EPQR-A, Big Five, etc.).
    """

    __tablename__ = "questionnaire"
    __table_args__ = get_table_args()

    # Primary key (auto-increment)
    id: int | None = Field(
        default=None,
        primary_key=True,
        sa_column_kwargs={"autoincrement": True},
        description="Auto-increment primary key",
    )

    # Question identification
    language_questionnaire: Optional[str] = Field(
        default=None, max_length=50, index=True, description="Language of the questionnaire"
    )
    question_number: Optional[int] = Field(default=None, index=True, description="Question number")
    questionnaire: str = Field(default="epqra", description="Type of questionnaire")

    # Question details
    category: Optional[str] = Field(
        default=None, max_length=50, description="Personality category (N, E, P, L, etc.)"
    )
    key: Optional[int] = Field(default=None, description="Key value for scoring")
    question: Optional[str] = Field(default=None, description="Full text of the question")
    reverse: Optional[bool] = Field(
        default=False, description="Whether this is a reverse-scored question"
    )

# The Personality Trap Dataset

## Overview

This dataset contains the complete research data from "The Personality Trap" study, investigating demographic bias in Large Language Model (LLM) persona generation through personality-driven prompting. The dataset includes 82,600 AI-generated personas across 5 different LLMs, along with their psychological profiles and demographic characteristics.

**Research Paper**: "The Personality Trap: How LLMs Embed Bias When Generating Human-Like Personas" (under review)

**Dataset Repository**: [https://zenodo.org/records/17286313](https://zenodo.org/records/17286313) | DOI: [10.5281/zenodo.17286313](https://doi.org/10.5281/zenodo.17286313)

**Code Repository**: [https://github.com/gregorio-ferreira/the_personality_trap](https://github.com/gregorio-ferreira/the_personality_trap)

**Dataset Size**: ~47MB (PostgreSQL dump, gzipped)  
**Total Records**: 698,632 across 9 tables + 1 materialized view  
**Personas Generated**: 82,600 unique personas  
**Models Tested**: 5 (GPT-3.5, GPT-4o, Claude 3.5 Sonnet, Llama 3.1 70B, Llama 3.2 3B)  
**Personality Profiles**: 826 unique baseline personalities (EPQR-A)  
**Experimental Conditions**: Baseline, MaxN (high neuroticism), MaxP (high psychoticism)  
**Backup Location**: `dataset/20251008/` (Complete database backup and CSV exports)

## Research Context

This dataset was created to study demographic bias in AI persona generation when LLMs are conditioned on different personality traits. The research explores whether personality-driven prompting can reveal or mitigate demographic biases in AI-generated personas, particularly examining how extreme personality conditions (borderline traits) affect the demographic distributions of generated characters.

### Key Research Questions
1. Do LLMs exhibit demographic bias when generating personas based on personality traits?
2. How do extreme personality conditions (high neuroticism, high psychoticism) affect demographic representation?
3. Are there systematic differences in bias patterns across different LLM providers and models?

## Dataset Structure

The dataset consists of 9 tables plus 1 materialized view in PostgreSQL schema `personality_trap`:

### Core Tables (9)
1. **personas** - AI-generated personas with demographics (82,600 records)
2. **reference_questionnaires** - Personality questionnaire responses (19,824 records)
3. **random_questionnaires** - Random baseline questionnaires (19,824 records)
4. **experiments_list** - Individual experiment records (17,346 records)
5. **experiments_groups** - Experiment configurations (22 records)
6. **eval_questionnaires** - Parsed LLM questionnaire responses (541,680 records)
7. **questionnaire** - Master questionnaire definitions (752 records)
8. **experiment_request_metadata** - Raw API request/response data (18,573 records)
9. **alembic_version** - Database migration tracking (1 record)

### Materialized View (1)
- **experiments_evals** - Pre-computed questionnaire evaluations with cleaned model/population mappings (optimized for analysis queries)

For complete database restoration instructions, see [`docs/DATABASE_BACKUP.md`](docs/DATABASE_BACKUP.md).

### 1. `personas` (82,600 records)
**Primary table containing all generated AI personas with demographic and descriptive information.**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | integer | Primary key, auto-generated | 1 |
| `ref_personality_id` | integer | Reference to original personality profile (1-826) | 1 |
| `population` | varchar(100) | Generation condition and model | `borderline_maxN_gpt4o` |
| `model` | varchar(100) | LLM model used | `gpt4o`, `claude35sonnet` |
| `name` | varchar(255) | Generated persona name | `Emily Chen` |
| `age` | integer | Generated age | 28 |
| `gender` | varchar(20) | Cleaned gender category | `female`, `male`, `non-binary` |
| `gender_original` | varchar(50) | Original gender as generated | `Female` |
| `race` | varchar(50) | Cleaned race category | `Asian`, `White`, `Black`, `Hispanic` |
| `race_original` | varchar(100) | Original race as generated | `Asian American` |
| `sexual_orientation` | varchar(20) | Cleaned sexual orientation | `heterosexual`, `lgbtq+` |
| `sexual_orientation_original` | varchar(100) | Original sexual orientation | `Heterosexual` |
| `ethnicity` | varchar(100) | Ethnicity information | `Chinese American` |
| `religious_belief` | varchar(50) | Cleaned religious belief | `Christian`, `Muslim`, `None` |
| `religious_belief_original` | varchar(100) | Original religious belief | `Non-denominational Christian` |
| `occupation` | varchar(100) | Cleaned occupation category | `Accounting & finance` |
| `occupation_original` | varchar(255) | Original occupation | `Senior Financial Analyst` |
| `political_orientation` | varchar(20) | Cleaned political orientation | `Progressive`, `Conservative` |
| `political_orientation_original` | varchar(50) | Original political orientation | `Moderate liberal` |
| `location` | varchar(100) | Cleaned location | `Seattle, WA` |
| `location_original` | varchar(255) | Original location | `Seattle, Washington, USA` |
| `description` | text | Full persona narrative/description | Full persona backstory |
| `word_count_description` | integer | Word count of description | 245 |
| `repetitions` | integer | Number of generations for this configuration | 1 |
| `model_print` | varchar(50) | Cleaned model name for display | `GPT-4o`, `Claude-3.5-s` |
| `population_print` | varchar(50) | Cleaned population name for display | `Base`, `Max N`, `Max P` |

**Population Types:**
- `generated_{model}_spain826`: Baseline personas (normal personality traits)
- `borderline_maxN_{model}`: High neuroticism condition
- `borderline_maxP_{model}`: High psychoticism condition

**Models:**
- `gpt35`: GPT-3.5 Turbo
- `gpt4o`: GPT-4o
- `claude35sonnet`: Claude 3.5 Sonnet
- `llama3170B`: Llama 3.1 70B
- `llama323B`: Llama 3.2 3B

### 2. `reference_questionnaires` (19,824 records)
**EPQR-A (Eysenck Personality Questionnaire Revised-Abbreviated) responses used as input for persona generation.**

This table contains the 826 unique personality profiles that served as the foundation for generating AI personas. Each profile consists of 24 questions from the EPQR-A questionnaire, measuring four personality dimensions.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | integer | Primary key | 1 |
| `personality_id` | integer | Personality profile ID (1-826) | 1 |
| `question_number` | integer | Question number (1-24) | 1 |
| `question` | text | Full text of the personality question | `Does your mood often go up and down?` |
| `category` | varchar(10) | EPQR-A dimension | `N`, `E`, `P`, `L` |
| `key` | boolean | Whether this is a key question for scoring | `true` |
| `answer` | boolean | True/False response to the question | `true` |

**EPQR-A Dimensions:**
- **N (Neuroticism)**: Emotional stability vs. instability (6 questions)
- **E (Extraversion)**: Social energy and outgoingness (6 questions)
- **P (Psychoticism)**: Unconventional thinking and tough-mindedness (6 questions)
- **L (Lie Scale)**: Social desirability and response validity (6 questions)

**Answer Distribution** (across 826 personalities):
- Neuroticism: 51.3% Yes
- Extraversion: 44.5% Yes
- Psychoticism: 50.7% Yes
- Lie Scale: 17.0% Yes

**Purpose**: These 826 personality profiles represent the baseline personalities from which all personas were generated. The profiles were sourced from a representative Spanish population sample and used to create personality-conditioned prompts for LLMs.

### 3. `random_questionnaires` (19,824 records)
**Randomly generated EPQR-A questionnaire responses for control comparison.**

This table contains randomly generated personality questionnaire responses that serve as a control baseline for comparing against the reference questionnaires. The random responses provide a null hypothesis baseline: what would personality distributions look like if answers were generated purely at random?

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | integer | Primary key | 1 |
| `personality_id` | integer | Personality profile ID (1-826) | 1 |
| `question_number` | integer | Question number (1-24) | 1 |
| `question` | text | Full text of the personality question | `Does your mood often go up and down?` |
| `category` | varchar(10) | EPQR-A dimension | `N`, `E`, `P`, `L` |
| `key` | boolean | Whether this is a key question for scoring | `true` |
| `answer` | boolean | Randomly generated True/False response | `true` |

**Random Generation Method:**
For each question, answers were randomly generated using the empirical probability distribution of Yes/No responses from the reference questionnaires:
- **Neuroticism (N)**: ~50% Yes, ~50% No
- **Extraversion (E)**: ~44% Yes, ~56% No
- **Psychoticism (P)**: ~51% Yes, ~49% No
- **Lie Scale (L)**: ~17% Yes, ~83% No

This approach ensures the random baseline matches the overall response patterns of the reference population while removing any systematic personality-demographic associations.

**Actual Distribution** (in random questionnaires):
- Neuroticism: 50.2% Yes
- Extraversion: 44.2% Yes
- Psychoticism: 50.6% Yes
- Lie Scale: 16.6% Yes

**Purpose:**
- **Control Baseline**: Provides a random baseline for statistical comparison against the reference questionnaires
- **Bias Detection**: Helps identify systematic patterns in persona generation that differ from random chance
- **Statistical Testing**: Enables null hypothesis testing for personality-demographic associations
- **Same Structure**: Identical schema to `reference_questionnaires` to enable direct comparison

**EPQR-A Dimensions** (same as reference_questionnaires):
- **N (Neuroticism)**: Emotional stability vs. instability
- **E (Extraversion)**: Social energy and outgoingness
- **P (Psychoticism)**: Unconventional thinking and tough-mindedness
- **L (Lie Scale)**: Social desirability and response validity

### 4. `experiments_list` (17,346 records)
**Individual experiment records tracking each persona generation attempt.**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `experiment_id` | integer | Primary key | 1 |
| `experiments_group_id` | integer | Reference to experiment group | 1 |
| `repeated` | integer | Repetition number | 1 |
| `created` | varchar | Timestamp of experiment | `2025-03-12 10:30:00` |
| `questionnaire` | varchar | Questionnaire type used | `epqra` |
| `language_instructions` | varchar | Language for instructions | `en` |
| `language_questionnaire` | varchar | Language for questionnaire | `en` |
| `model_provider` | varchar | API provider | `openai` |
| `model` | varchar | LLM model used | `gpt-4o` |
| `population` | varchar | Population/condition identifier | `borderline_maxN` |
| `personality_id` | integer | Reference to personality profile | 1 |
| `succeeded` | boolean | Whether generation succeeded | `true` |
| `repo_sha` | varchar | Git commit SHA | `abc123...` |
| `llm_explanation` | text | LLM's explanation of questionnaire responses | `I answered based on my personality...` |

### 5. `experiments_groups` (22 records)
**Experiment group configurations organizing related experiments.**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `experiments_group_id` | integer | Primary key | 1 |
| `description` | text | Description of experiment group | `Baseline persona generation` |
| `system_role` | text | System role prompt used | LLM system instructions |
| `base_prompt` | text | Base prompt template | Persona generation template |
| `temperature` | float | LLM temperature parameter | 0.7 |
| `top_p` | float | Top-p sampling parameter | 1.0 |
| `concluded` | boolean | Whether group is completed | `true` |
| `processed` | boolean | Whether results are processed | `true` |
| `created` | varchar | Creation timestamp | `2025-03-12` |

### 6. `eval_questionnaires` (541,680 records)
**Parsed questionnaire responses from LLMs for each experiment.**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | integer | Primary key | 1 |
| `experiment_id` | integer | Reference to experiment | 1 |
| `question_number` | integer | Question number (1-24) | 1 |
| `answer` | integer | Parsed answer value | 1 |

### 7. `questionnaire` (752 records)
**Master questionnaire definitions with questions in multiple languages.**

This table contains the complete question bank for both EPQR-A and Big Five personality questionnaires used in the experiments.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | integer | Primary key | 1 |
| `language_questionnaire` | varchar(50) | Language code | `en`, `es`, `pt`, `de`, `fr`, `it` |
| `question_number` | integer | Question number | 1 |
| `questionnaire` | varchar | Questionnaire type | `epqra`, `bigfive` |
| `category` | varchar(50) | Personality dimension | `N`, `E`, `P`, `L` (EPQR-A) or Big Five |
| `key` | integer | Scoring key | 1 |
| `question` | text | Question text | `Does your mood often go up and down?` |
| `reverse` | boolean | Whether reverse scored | `false` |

**Questionnaire Types:**
- **EPQR-A**: 24 questions in 6 languages (576 records)
  - Used for primary persona generation
  - Categories: N (Neuroticism), E (Extraversion), P (Psychoticism), L (Lie Scale)
- **Big Five**: 44 questions in English only (176 records)
  - Used for validation experiments
  - Categories: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism

**Languages Available:**
- English (en): EPQR-A (24 questions) + Big Five (44 questions)
- Spanish (es), Portuguese (pt), German (de), French (fr), Italian (it): EPQR-A only (24 questions each)

### 8. `experiment_request_metadata` (18,573 records)
**Raw request and response data from LLM API calls.**

This table stores the complete request/response cycle for each experiment, including the full prompts sent to LLMs, the raw responses received, and metadata about token usage and API parameters.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | integer | Primary key | 1 |
| `experiment_id` | integer | Reference to experiment | 1 |
| `request` | text | Full prompt/request sent to LLM | `Generate a persona with...` |
| `response` | text | Raw LLM response | `{"name": "John Doe"...}` |
| `request_metadata` | jsonb | Request parameters (model, temperature, etc.) | `{"model": "gpt-4o", "temperature": 0.7}` |
| `response_metadata` | jsonb | Response metadata (tokens, timing, etc.) | `{"tokens": 450, "finish_reason": "stop"}` |

**Purpose:**
- **Reproducibility**: Complete record of all LLM interactions for research reproducibility
- **Token Usage Tracking**: Monitor API costs and usage patterns across experiments
- **Prompt Engineering Analysis**: Analyze how different prompts affect persona generation
- **Error Debugging**: Investigate failed or problematic generations
- **Model Comparison**: Compare request/response patterns across different LLM providers

**JSON Metadata Fields:**
- `request_metadata`: Model parameters, API version, provider-specific settings
- `response_metadata`: Token counts, finish reasons, latency metrics, error messages

### 9. `alembic_version` (1 record)
**Database migration version tracking (internal use).**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `version_num` | varchar(32) | Current migration version | `0001` |

**Important**: The `alembic_version` table is stored **inside** the `personality_trap` schema (not in the `public` schema). This enables multiple independent schemas to coexist in the same database, each with its own migration tracking.

### Materialized View: `experiments_evals`
**Pre-computed questionnaire evaluation results with cleaned mappings for analysis.**

This materialized view combines data from experiments_list, experiments_groups, eval_questionnaires, and questionnaire tables to provide optimized access to questionnaire evaluation data with pre-mapped model and population names.

**Key Features:**
- **Duplicate Handling**: Uses `DISTINCT ON` to ensure exactly one record per (experiment_id, question_number)
- **Pre-computed Evaluations**: Calculates scored responses based on questionnaire type (EPQR-A vs Big Five)
- **Cleaned Mappings**: Includes `model_clean` and `population_mapped` columns for consistent analysis
- **English Only**: Filters to English-language questionnaires only
- **Performance Optimized**: Indexed on common query patterns (model, population, experiment_id)

**Important Columns:**
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `experiment_id` | integer | Reference to experiment | 1 |
| `model` | varchar | Original model identifier | `gpt-4o-2024-11-20` |
| `model_clean` | varchar | Cleaned model name for display | `GPT-4o` |
| `population` | varchar | Original population identifier | `borderline_maxN_gpt4o` |
| `population_mapped` | varchar | Cleaned population name | `maxN_gpt4o` |
| `population_display` | varchar | Display name for tables | `MaxN` |
| `questionnaire` | varchar | Questionnaire type | `epqra`, `bigfive` |
| `category` | varchar | Personality dimension | `N`, `E`, `P`, `L` |
| `question_number` | integer | Question number (1-24) | 1 |
| `answer` | integer | Raw answer value | 1 |
| `eval` | integer | Scored evaluation value | 1 |

**Model Mapping:**
- `gpt-3.5-turbo-0125` → `GPT-3.5`
- `gpt-4o-2024-11-20` → `GPT-4o`
- `anthropic.claude-3-5-sonnet-20240620-v1:0` → `Claude-3.5-s`
- `us.meta.llama3-1-70b-instruct-v1:0` → `Llama3.1-70B`
- `eu.meta.llama3-2-3b-instruct-v1:0` → `Llama3.2-3B`

**Population Display Mapping:**
- Base models (e.g., `gpt4o`, `claude35sonnet`) → `Base`
- MaxN variants (e.g., `maxN_gpt4o`) → `MaxN`
- MaxP variants (e.g., `maxP_gpt4o`) → `MaxP`

**Setup:**
The materialized view is created using the SQL script in `dataset/20251008/sql/create_experiments_evals_view.sql`. It must be manually created after Alembic migrations since Alembic does not handle materialized views. See restoration instructions in [`docs/DATABASE_BACKUP.md`](docs/DATABASE_BACKUP.md).

**Refresh:**
To refresh the materialized view with updated data:
```sql
REFRESH MATERIALIZED VIEW personality_trap.experiments_evals;
```

**Indexes:**
The view includes performance indexes on:
- `experiment_id`
- `model_clean`
- `population_mapped`
- `(model_clean, population_mapped)` - composite index
- `(questionnaire, category)`
- `(questionnaire, model_clean, population_mapped)` - composite index

## Data Collection Methodology

### Personality Profile Generation
1. **Base Personalities**: 826 unique personality profiles were sourced from a representative Spanish population sample
2. **Questionnaire**: EPQR-A (Eysenck Personality Questionnaire Revised-Abbreviated) with 24 questions across 4 dimensions:
   - **Neuroticism (N)**: 6 questions measuring emotional stability
   - **Extraversion (E)**: 6 questions measuring social energy
   - **Psychoticism (P)**: 6 questions measuring unconventional thinking
   - **Lie Scale (L)**: 6 questions measuring social desirability bias
3. **Borderline Conditions**: Modified versions with extreme personality traits:
   - **MaxN (High Neuroticism)**: All 6 neuroticism questions answered to maximize anxiety/emotional instability
   - **MaxP (High Psychoticism)**: All 6 psychoticism questions answered to maximize unconventional thinking/tough-mindedness
4. **Random Baseline**: 826 randomly generated personality profiles using empirical probability distributions from reference questionnaires to serve as null hypothesis control

### LLM Persona Generation
1. **Models Tested**: 
   - OpenAI: GPT-3.5 Turbo, GPT-4o
   - Anthropic: Claude 3.5 Sonnet
   - Meta: Llama 3.1 70B, Llama 3.2 3B
2. **Prompt Engineering**: Personality-driven prompts asking LLMs to generate realistic personas consistent with given EPQR-A personality traits
3. **Output Format**: Structured JSON responses with demographic fields (gender, race, age, sexual orientation, religious belief, political orientation, occupation, location), name, and narrative description
4. **Quality Control**: Multi-stage automatic parsing and validation of generated responses
5. **Experiments**: Two questionnaire types used to validate LLM persona consistency:
   - **EPQR-A**: 12,390 experiments (primary)
   - **Big Five**: 4,956 experiments (validation)

### Data Processing
1. **Demographic Cleaning**: Original free-text demographic responses were categorized into standardized classifications
   - Gender: {male, female, non-binary}
   - Race: {White, Black, Asian, Hispanic, Other}
   - Sexual Orientation: {heterosexual, lgbtq+}
   - Religion: {Christian, Muslim, Jewish, Hindu, Buddhist, None, Other}
   - Political Orientation: {Progressive, Conservative, Moderate, Libertarian, Other}
2. **Validation**: Multi-step validation process for JSON parsing and demographic classification consistency
3. **Deduplication**: Systematic handling of repeated generations and edge cases
4. **Materialized View Creation**: Pre-computed evaluation metrics with cleaned model and population mappings for optimized analysis queries

## Research Applications

This dataset enables research in several key areas:

### Bias Analysis
- **Demographic Distribution Analysis**: Compare representation of different demographic groups across models and conditions
- **Intersectionality Studies**: Examine how multiple demographic characteristics interact in AI-generated personas
- **Model Comparison**: Systematic comparison of bias patterns across different LLM providers

### Personality Psychology
- **Trait-Demographic Associations**: Analyze how personality traits correlate with generated demographic characteristics
- **Cross-Cultural Validation**: Examine personality expression across different demographic groups
- **Borderline Personality Effects**: Study how extreme personality conditions affect persona generation

### AI Ethics and Fairness
- **Representation Studies**: Quantify over/under-representation of demographic groups
- **Bias Mitigation**: Test effectiveness of personality-driven prompting as bias mitigation strategy
- **Model Auditing**: Comprehensive bias assessment across multiple LLM providers

## Data Quality and Limitations

### Strengths
- **Large Scale**: 82,600 personas across 5 major LLMs
- **Systematic Design**: Controlled experimental conditions with consistent methodology
- **Rich Metadata**: Comprehensive experimental tracking and demographic categorization
- **Multiple Conditions**: Baseline and extreme personality conditions for comparison

### Limitations
- **English Only**: All prompts and responses in English
- **Western Context**: Personality framework and demographic categories reflect Western contexts
- **Binary Classifications**: Some demographic categories use simplified binary or limited categorical classifications
- **Temporal Snapshot**: Data reflects LLM capabilities and biases as of March 2025
- **Synthetic Data**: All personas are AI-generated, not based on real individuals

### Recommended Quality Checks
1. **Demographic Distribution Analysis**: Check for unexpected patterns in demographic distributions
2. **Response Validation**: Verify JSON parsing accuracy and demographic classification consistency
3. **Model Performance**: Analyze success rates and parsing quality across different models
4. **Personality Trait Validation**: Confirm that borderline conditions produced expected personality modifications

## Usage Guidelines

### Ethical Considerations
- **Synthetic Data Only**: All personas are fictional AI-generated characters, not based on real individuals
- **Bias Research**: Dataset contains and reflects existing biases in LLMs - use for bias research and fairness evaluation, not perpetuation
- **Cultural Sensitivity**: Be aware of Western-centric personality frameworks (EPQR-A) and demographic categories
- **Academic Use**: Intended primarily for academic research on AI bias, fairness, and demographic representation
- **Responsible Disclosure**: When reporting findings, contextualize within broader AI ethics and fairness research

### Technical Requirements
- **Database**: PostgreSQL 14+ recommended (tested with PostgreSQL 16)
- **Storage**: ~50MB for compressed dataset, ~280MB uncompressed
- **Analysis Tools**: 
  - Python 3.12+ with pandas, numpy, scipy, pingouin (see `pyproject.toml` in code repository)
  - R with tidyverse for alternative analysis workflows
  - Jupyter notebooks for interactive exploration (provided in code repository)
- **Statistical Software**: Python statistical libraries (statsmodels, pingouin) or R for demographic and bias analysis
- **Replication Package**: Full code repository at [https://github.com/gregorio-ferreira/the_personality_trap](https://github.com/gregorio-ferreira/the_personality_trap)

### Recommended Workflow

1. **Setup Environment**: Clone code repository and follow setup instructions in `README.md`
2. **Restore Dataset**: Use automated restoration script (`dataset/20251008/restore_backup.sh`) or Make commands
3. **Explore Data**: Use provided Jupyter notebooks in `examples/` directory to explore dataset structure
4. **Reproduce Results**: Run analysis notebooks to reproduce paper figures and tables
5. **Extend Analysis**: Modify scripts or create new analyses using the complete dataset

For detailed instructions, see the [code repository](https://github.com/gregorio-ferreira/the_personality_trap).

## Database Schema SQL

The complete database schema can be recreated using the following approaches:

### Option 1: PostgreSQL Full Restore (Recommended)
Use the complete database dump for fastest restoration:
```bash
# Restore from compressed dump
gunzip -c dataset/20251008/sql/personas_database_backup.sql.gz | psql -U postgres -d personas
```

### Option 2: Alembic Migrations (Development)
For development environments, use Alembic migrations for version-controlled schema setup:
```bash
# Apply all migrations
make db-upgrade

# Manually create materialized view (not handled by Alembic)
psql -U postgres -d personas -f dataset/20251008/sql/create_experiments_evals_view.sql
```

### Option 3: CSV Import (Analysis Only)
For data analysis without full database setup, CSV files are available:
```bash
# CSVs located in dataset/20251008/csv/
# See dataset/20251008/scripts/ for import scripts
```

See [`docs/DATABASE_BACKUP.md`](docs/DATABASE_BACKUP.md) for complete restoration instructions.

## Backup Structure

Complete dataset backup is available in `dataset/20251008/`:

```
dataset/20251008/
├── README.md                           # Complete setup and restoration guide
├── backup_manifest.txt                 # Detailed file inventory and table counts
├── csv/                                # CSV exports (all 9 tables)
│   ├── personas.csv                    # 82,600 persona records
│   ├── reference_questionnaires.csv    # 19,824 personality responses
│   ├── random_questionnaires.csv       # 19,824 random baseline responses
│   ├── experiments_list.csv            # 17,346 experiment records
│   ├── experiments_groups.csv          # 22 experiment group configs
│   ├── eval_questionnaires.csv         # 541,680 LLM questionnaire responses
│   ├── questionnaire.csv               # 752 questionnaire definitions
│   └── experiment_request_metadata.csv # 18,573 API request/response records
├── sql/                                # SQL schema and data dumps
│   ├── personas_database_backup.sql.gz # Complete PostgreSQL dump (compressed)
│   ├── schema_export.sql               # Schema-only export
│   └── create_experiments_evals_view.sql # Materialized view creation script
├── alembic/                            # Alembic migration files (version-controlled)
│   └── versions/
│       └── 0001_create_complete_schema.py  # Single consolidated migration
├── scripts/                            # Backup and restoration utilities
│   ├── backup_script.py                # Generate backup from database
│   ├── restore_from_csv.py             # Restore from CSV files
│   └── verify_backup.py                # Verify backup integrity
└── docs/                               # Additional documentation
    ├── database_schema.md              # Detailed schema documentation
    └── restoration_guide.md            # Step-by-step restoration guide
```

### Backup Metadata
- **Backup Date**: October 8, 2025
- **Database Version**: PostgreSQL 16.10
- **Schema**: `personality_trap`
- **Alembic Version**: `001_initial_schema`
- **Total Size**: ~47MB (compressed), ~280MB (CSV exports)
- **Format**: PostgreSQL dump + CSV exports + restoration scripts

**Note**: Earlier dataset versions (20250928, 20251007) are also available in the `dataset/` directory but may not reflect the final published data.

## Contact and Support

### Code Repository

The complete replication package, including data processing pipelines, analysis scripts, and evaluation notebooks, is available at:

**GitHub**: [https://github.com/gregorio-ferreira/the_personality_trap](https://github.com/gregorio-ferreira/the_personality_trap)

The repository includes:
- Complete source code for persona generation pipeline
- LLM connector implementations (OpenAI, Bedrock, local models)
- Analysis and visualization scripts
- Jupyter notebooks reproducing all paper figures and tables
- Database schema and migration files
- Docker Compose setup for local development

### Dataset Repository

**Zenodo**: [https://zenodo.org/](https://zenodo.org/) (DOI to be assigned upon publication)

The dataset repository on Zenodo contains:
- Complete PostgreSQL database dump (47MB compressed)
- CSV exports of all tables (280MB)
- Database schema documentation
- Restoration scripts and guides
- This dataset description document

### Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{,
  title={The Personality Trap Dataset},
  author={},
  year={2025},
  publisher={Zenodo},
  doi={10.5281/zenodo.17286313},
  url={https://zenodo.org/records/17286313}
}
```

### Support and Documentation

For questions, issues, or additional information:

- **Dataset Restoration**: See [`docs/DATABASE_BACKUP.md`](docs/DATABASE_BACKUP.md) in the code repository
- **Usage Guide**: See [`docs/USAGE.md`](docs/USAGE.md) in the code repository
- **Troubleshooting**: See [`docs/DATABASE_BACKUP.md#troubleshooting`](docs/DATABASE_BACKUP.md#troubleshooting)
- **GitHub Issues**: [https://github.com/gregorio-ferreira/the_personality_trap/issues](https://github.com/gregorio-ferreira/the_personality_trap/issues)
- **Code Examples**: See `examples/` directory in the code repository

### Related Resources

- **Complete Backup**: `dataset/20251008/` - Database dumps, CSV files, restoration scripts
- **Schema Documentation**: [`examples/DATABASE_SCHEMA.md`](examples/DATABASE_SCHEMA.md) in code repository
- **Analysis Notebooks**: `examples/*.ipynb` in code repository (reproduces all paper results)
- **Verification Guide**: [`docs/RESTORATION_VERIFICATION.md`](docs/RESTORATION_VERIFICATION.md) in code repository

## Quick Access

- **Complete Backup**: `dataset/20251008/` - Database dumps, CSV files, migration scripts
- **Analysis Code**: `src/evaluations/` - Statistical analysis modules
- **Example Notebooks**: `examples/` - Research analysis notebooks
- **Database Models**: `src/personas_backend/db/models.py` - SQLModel definitions

## Changelog

- **v1.0**: Initial release with 82,600 personas across 5 LLMs
- **Future versions**: May include additional models, languages, or personality frameworks

---

**Keywords**: artificial intelligence, demographic bias, personality psychology, large language models, AI fairness, synthetic personas, LLM evaluation, bias mitigation

**Subject Areas**: Computer Science, Psychology, AI Ethics, Machine Learning, Social Computing

# Database Schema Reference - The Personality Trap

This document describes the database schema used in the research notebooks.

## Schema Overview

### Two-Schema Architecture

1. **Production Schema** (`personality_trap`)
   - Original research data (READ-ONLY)
   - Contains reference questionnaires and published results
   - Never modified by replication workflows

2. **Experimental Schema** (user-defined, e.g., `my_experiment`)
   - Created via Alembic migrations
   - Isolated workspace for replication
   - Safe to modify, reset, or delete

## Core Tables

### 1. `personas`
Stores AI-generated persona descriptions.

**Primary use:** Persona generation workflow

**Key columns:**
- `id` (PK): Auto-generated persona identifier
- `ref_personality_id`: Links to reference personality in questionnaires
- `population`: Population tag (e.g., `generated_gpt4o_spain826`)
- `model`: LLM model used for generation (e.g., `gpt4o`)
- `name`: Generated persona name
- `age`: Persona age
- `gender`, `race`, `sexual_orientation`: Demographics (cleaned)
- `gender_original`, `race_original`: Raw LLM output before cleaning
- `occupation`: Persona occupation
- `description`: Full persona narrative
- `created`: Timestamp

**Indexes:**
- `ref_personality_id`, `population`, `model`, `gender`, `race`

**Example query:**
```sql
SELECT model, population, COUNT(*)
FROM personas
GROUP BY model, population;
```

### 2. `reference_questionnaires`
Personality questionnaire responses from reference population.

**Primary use:** Input data for persona generation

**Key columns:**
- `personality_id`: Unique identifier for reference personality
- `question_number`: Question position (1-50 for Big Five)
- `question`: Question text
- `category`: Personality dimension (e.g., 'Openness', 'Neuroticism')
- `key`: Scoring key ('+' for positive, '-' for reverse-scored)
- `answer`: Numeric response (1-5 scale)

**Example query:**
```sql
SELECT personality_id, question_number, answer
FROM reference_questionnaires
WHERE personality_id = 1
ORDER BY question_number;
```

### 3. `experiments_groups`
Logical containers for batches of related experiments.

**Primary use:** Organize experiments by configuration

**Key columns:**
- `experiments_group_id` (PK): Auto-generated group identifier
- `created`: Timestamp
- `description`: Human-readable description
- `system_role`: Template for LLM system prompt
- `base_prompt`: Base instruction for questionnaire
- `temperature`, `top_p`: LLM sampling parameters
- `concluded`: Batch completion flag
- `processed`: Data processing flag

**Example query:**
```sql
SELECT experiments_group_id, description, created
FROM experiments_groups
ORDER BY created DESC;
```

### 4. `experiments_list`
Individual experiment records.

**Primary use:** Track questionnaire evaluation runs

**Key columns:**
- `experiment_id` (PK): Auto-generated experiment identifier
- `experiments_group_id` (FK): Links to parent group
- `questionnaire`: Questionnaire type ('bigfive', 'epqr_a')
- `model`: LLM model identifier
- `model_provider`: API provider (e.g., 'openai', 'bedrock')
- `population`: Persona population tag
- `personality_id`: Reference personality ID
- `repeated`: Repetition number (for multi-run experiments)
- `succeeded`: Boolean success/failure flag
- `llm_explanation`: Error messages or notes
- `repo_sha`: Git commit SHA for reproducibility
- `created`, `updated`: Timestamps

**Indexes:**
- `experiments_group_id`, `personality_id`

**Example query:**
```sql
SELECT model, population,
       COUNT(*) as total,
       SUM(CASE WHEN succeeded THEN 1 ELSE 0 END) as succeeded
FROM experiments_list
GROUP BY model, population;
```

### 5. `eval_questionnaires`
Questionnaire answers collected from LLMs.

**Primary use:** Store and analyze personality responses

**Key columns:**
- `id` (PK): Auto-generated answer identifier
- `experiment_id` (FK): Links to `experiments_list`
- `question_number`: Question position
- `answer`: Numeric response (1-5 scale)
- `created`: Timestamp

**Indexes:**
- `experiment_id`, `question_number`

**Example query:**
```sql
SELECT question_number, answer
FROM eval_questionnaires
WHERE experiment_id = 123
ORDER BY question_number;
```

**Analysis example:**
```sql
-- Calculate average answer by question for an experiment group
SELECT
    eq.question_number,
    AVG(eq.answer) as avg_answer,
    STDDEV(eq.answer) as std_answer
FROM eval_questionnaires eq
JOIN experiments_list el ON eq.experiment_id = el.experiment_id
WHERE el.experiments_group_id = 1
GROUP BY eq.question_number
ORDER BY eq.question_number;
```

### 6. `experiment_request_metadata`
LLM API request/response logs.

**Primary use:** Debugging, reproducibility, API usage tracking

**Key columns:**
- `id` (PK): Auto-generated metadata identifier
- `experiment_id` (FK): Links to `experiments_list`
- `request_json` (JSONB): Full LLM API request
- `response_json` (JSONB): Full LLM API response
- `request_metadata` (JSONB): Additional metadata (tokens, timing, etc.)
- `created`: Timestamp

**Indexes:**
- `experiment_id`

**Example query:**
```sql
-- Extract token usage from response metadata
SELECT
    experiment_id,
    response_json->'usage'->>'total_tokens' as total_tokens,
    response_json->'usage'->>'prompt_tokens' as prompt_tokens,
    response_json->'usage'->>'completion_tokens' as completion_tokens
FROM experiment_request_metadata
WHERE experiment_id IN (SELECT experiment_id FROM experiments_list WHERE model = 'gpt4o')
LIMIT 10;
```

## Relationships

```
experiments_groups (1) ──< (N) experiments_list
                                       │
                                       ├──< (N) eval_questionnaires
                                       │
                                       └──< (N) experiment_request_metadata

personas (N) ──── (1) reference_questionnaires
     │                      (via ref_personality_id)
     │
     └──── (N) experiments_list
                (via personality_id)
```

## Common Queries

### Count personas by model
```sql
SELECT model, COUNT(*) as count
FROM personas
GROUP BY model
ORDER BY count DESC;
```

### Find incomplete experiments
```sql
SELECT
    el.experiment_id,
    el.model,
    el.personality_id,
    COUNT(eq.id) as answers_count
FROM experiments_list el
LEFT JOIN eval_questionnaires eq ON el.experiment_id = eq.experiment_id
WHERE el.succeeded = TRUE
GROUP BY el.experiment_id, el.model, el.personality_id
HAVING COUNT(eq.id) < 50  -- Big Five has 50 questions
ORDER BY el.experiment_id;
```

### Calculate experiment success rate
```sql
SELECT
    model,
    population,
    COUNT(*) as total_experiments,
    SUM(CASE WHEN succeeded THEN 1 ELSE 0 END) as succeeded,
    ROUND(100.0 * SUM(CASE WHEN succeeded THEN 1 ELSE 0 END) / COUNT(*), 2) as success_rate
FROM experiments_list
GROUP BY model, population
ORDER BY model, population;
```

### Get experiment duration statistics
```sql
SELECT
    eq.experiment_id,
    MIN(eq.created) as start_time,
    MAX(eq.created) as end_time,
    MAX(eq.created) - MIN(eq.created) as duration
FROM eval_questionnaires eq
GROUP BY eq.experiment_id
ORDER BY duration DESC
LIMIT 10;
```

## Schema Migrations

Migrations are managed via Alembic:

```bash
# Apply migrations to create schema
make db-upgrade

# Create new migration (for schema changes)
make db-revision msg="description"

# Rollback last migration (development only)
make db-downgrade
```

## Data Integrity

### Constraints
- Primary keys on all tables
- Foreign keys linking experiments to groups
- Indexes on frequently queried columns

### Validation Rules
- `answer` values in `eval_questionnaires`: 1-5 range
- `succeeded` in `experiments_list`: boolean (TRUE/FALSE/NULL)
- `personality_id` must exist in reference data

### Recommended Checks

Before analysis, verify:

```sql
-- Check for duplicate experiment registrations
SELECT experiment_id, COUNT(*)
FROM experiments_list
GROUP BY experiment_id
HAVING COUNT(*) > 1;

-- Check for missing answers
SELECT el.experiment_id, el.questionnaire,
       COUNT(DISTINCT eq.question_number) as questions_answered
FROM experiments_list el
LEFT JOIN eval_questionnaires eq ON el.experiment_id = eq.experiment_id
WHERE el.succeeded = TRUE
GROUP BY el.experiment_id, el.questionnaire
HAVING COUNT(DISTINCT eq.question_number) <
    CASE el.questionnaire
        WHEN 'bigfive' THEN 50
        WHEN 'epqr_a' THEN 48
    END;
```

## Performance Considerations

### Recommended Indexes (already created)
- `personas`: `ref_personality_id`, `population`, `model`
- `experiments_list`: `experiments_group_id`, `personality_id`
- `eval_questionnaires`: `experiment_id`, `question_number`

### Query Optimization Tips
- Use `experiment_id` filters to limit result sets
- Join on indexed columns
- Use `LIMIT` for exploratory queries on large datasets
- Consider materialized views for complex aggregations

## Cleanup

To reset your experimental schema:

```sql
-- WARNING: This deletes all data in experimental schema
DROP SCHEMA my_experiment CASCADE;
```

Then re-run migrations:
```bash
make db-upgrade
```

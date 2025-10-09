# Example Notebooks - Research Replication Guide

This directory contains Jupyter notebooks demonstrating how to replicate the research from **"The Personality Trap: Evaluating Psychological Profiles in Large Language Models"**.

## Overview

The research workflow consists of two main phases:

1. **Persona Generation** (`personas_generation.ipynb`): Generate AI personas using LLMs
2. **Questionnaire Experiments** (`questionnaires_experiments.ipynb`): Evaluate personas with personality questionnaires

## Prerequisites

### Environment Setup

1. **Install dependencies:**
   ```bash
   # Using uv (recommended)
   uv sync

   # Or using pip
   pip install -e .
   ```

2. **Start PostgreSQL database:**
   ```bash
   make db-up  # Uses Docker Compose
   ```

3. **Configure API keys and database credentials:**
   - Copy `example_config.yaml` to your config location
   - Set environment variables or update YAML with:
     - PostgreSQL connection details
     - OpenAI API key (for GPT models)
     - AWS credentials (for Claude/Bedrock models)
     - Llama model endpoints (if using local models)

4. **Verify setup:**
   ```bash
   make status  # Check database connectivity and environment
   ```

## Workflow

### Phase 1: Persona Generation

**Notebook:** `personas_generation.ipynb`

**Purpose:** Generate diverse AI personas based on reference personality questionnaires.

**Steps:**
1. **Schema Setup**: Create experimental schema using Alembic migrations
2. **Reference Data**: Copy questionnaire responses from research data
3. **Baseline Personas**: Generate personas using multiple LLM models
4. **Borderline Personas** (optional): Create experimental personality variants
5. **Verification**: Validate generated personas in database

**Output:**
- `{schema}.personas` table populated with AI-generated personas
- Each persona includes: name, age, demographics, occupation, personality traits

**Time estimate:** 10-30 minutes (depends on number of models and personalities)

**Key configuration:**
```python
N_PERSONALITIES = 2  # Start with 2-5 for testing
baseline_models = [ModelID.GPT4O]  # Add more models as needed
REFERENCE_POPULATION = 'spain826'
```

### Phase 2: Questionnaire Experiments

**Notebook:** `questionnaires_experiments.ipynb`

**Purpose:** Administer personality questionnaires to LLMs impersonating the generated personas.

**Steps:**
1. **Verify Personas**: Confirm persona generation completed successfully
2. **Register Experiments**: Create experiment group and individual experiment records
3. **Execute Experiments**: LLMs answer questionnaires while impersonating personas
4. **Verify Results**: Check experiment status and answer completeness
5. **Inspect Data**: Review questionnaire answers and API metadata

**Output:**
- `{schema}.experiments_list` table with experiment records
- `{schema}.eval_questionnaires` table with questionnaire answers
- `{schema}.experiment_request_metadata` table with LLM API logs

**Time estimate:** 15-60 minutes (depends on number of experiments and API latency)

**Key configuration:**
```python
QUESTIONNAIRE_TYPE = "bigfive"  # or "epqr_a"
EXPERIMENT_MODEL = ModelID.GPT4O
REPETITIONS = 1
```

## Database Schema Architecture

The system uses two schemas:

### Production Schema (`personality_trap`)
- Contains original research data
- **READ-ONLY** - never modified by notebooks
- Tables: `reference_questionnaires`, original research results

### Experimental Schema (configured via `schema.target_schema`)
- Your working area for replication
- Created fresh by Alembic migrations
- Isolated from production data

**Tables created:**
- `personas`: Generated AI personas with demographics
- `reference_questionnaires`: Personality questionnaire responses (copied from research)
- `experiments_groups`: Logical containers for related experiments
- `experiments_list`: Individual experiment runs
- `eval_questionnaires`: Questionnaire answers from LLMs
- `experiment_request_metadata`: LLM API request/response logs

## Configuration

### Schema Configuration

Edit your `.yaml` config file:

```yaml
schema:
  default_schema: "personality_trap"  # Production (read-only)
  target_schema: "my_experiment"      # Your experimental schema
```

Or use environment variables:
```bash
export PERSONAS_TARGET_SCHEMA="my_experiment"
```

### Model Configuration

The notebooks support multiple LLM providers:

- **OpenAI**: GPT-3.5, GPT-4o (requires `OPENAI_API_KEY`)
- **AWS Bedrock**: Claude 3.5 Sonnet (requires AWS credentials)
- **Local/Hosted**: Llama models (configure endpoints in YAML)

## Common Issues and Troubleshooting

### "No personas found" error
**Cause:** `personas_generation.ipynb` not run or failed
**Solution:** Complete Phase 1 notebook first

### "Schema does not exist" error
**Cause:** Migrations not applied
**Solution:** Run the schema setup cell in `personas_generation.ipynb`

### API rate limit errors
**Cause:** Too many concurrent requests
**Solution:** Reduce `max_workers` parameter (try 1-2 instead of 3)

### Missing questionnaire answers
**Cause:** LLM response parsing failures
**Solution:** Check `experiment_request_metadata` table for raw responses, verify LLM output format

### Database connection errors
**Cause:** PostgreSQL not running or wrong credentials
**Solution:**
```bash
make db-up  # Start database
make status # Verify connection
```

## Data Export

Export results for analysis:

```python
# From personas_generation.ipynb
generated_personas_df.to_csv('personas.csv', index=False)

# From questionnaires_experiments.ipynb
answers_df.to_csv('questionnaire_answers.csv', index=False)
experiment_status.to_csv('experiment_status.csv', index=False)
```

## Research Replication Notes

### Full Replication
To replicate the complete research:
1. Use all available models: GPT-3.5, GPT-4o, Claude 3.5 Sonnet, Llama 3.2 3B, Llama 3.1 70B
2. Generate both baseline and borderline personas
3. Run experiments with both questionnaire types: Big Five and EPQ-R-A
4. Use multiple repetitions (3-5) for statistical validity

### Minimal Replication (for testing)
1. Use single model: GPT-4o
2. Generate baseline personas only (skip borderline)
3. Test with 2-5 reference personalities
4. Single repetition per experiment

### Performance Optimization
- **Concurrent workers**: Adjust based on API rate limits
  - OpenAI: 3-5 workers typical
  - AWS Bedrock: 2-3 workers recommended
  - Local models: 1-2 workers to avoid overload

- **Batch processing**: For large-scale replication, process in batches
  - Set `batch_size` parameter to process incrementally
  - Monitor API quotas and costs

## Citation

If you use these notebooks for your research, please cite:

**Dataset Citation:**
```bibtex
@dataset{,
  title={},
  author={},
  year={2025},
  publisher={Zenodo},
  doi={[DOI TBD]},
  url={https://zenodo.org/[record-id]}
}
```

**Paper Citation:**
```bibtex
@article{,
  title={The Personality Trap: How LLMs Embed Bias When Generating Human-Like Personas},
  author={},
  journal={[Journal TBD]},
  year={2025},
  note={Under review}
}
```
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the notebook markdown cells for detailed explanations
3. Inspect database tables directly using SQL queries
4. Check logs in the `logs/` directory

## License

See [LICENSE](../LICENSE) in the repository root.

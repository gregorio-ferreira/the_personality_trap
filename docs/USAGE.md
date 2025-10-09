# Usage & Replication Guide

This document walks through the full workflow required to reproduce the experiments described in **The Personality Trap**.
Follow the steps below to provision the environment, populate the database, and execute the research workflow using Jupyter notebooks.

## 1. Install System Dependencies

The quickest path is to use the `make setup` target, which installs uv, synchronizes Python dependencies, and validates
PostgreSQL tooling.

```bash
make setup
```

If you prefer a manual setup, install uv and synchronize the environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --all-extras
```

## 2. Configure Credentials & Services

1. Copy the example configuration and fill in provider credentials:
   ```bash
   cp examples/example_config.yaml config.yaml
   ```
2. Export the required environment variables for LLM access and database connectivity (see [`README.md`](../README.md) for the
   complete list).
3. Ensure that PostgreSQL is accessible. You can start a local instance with Docker using `make db-up`.

## 3. Initialize or Restore the Database

Choose one of the following approaches:

- **Restore the published dataset (recommended for replication):**
  ```bash
  make db-upgrade    # Applies consolidated migration
  make db-restore    # Restores dataset/20251008/sql/personas_database_backup.sql.gz
  ```
  Detailed options and validation queries live in [`DATABASE_BACKUP.md`](DATABASE_BACKUP.md).

- **Run migrations on an empty database (for fresh experiments):**
  ```bash
  make db-upgrade
  ```

## 4. Run the Research Workflow via Notebooks

The research workflow is orchestrated through Jupyter notebooks in the `examples/` directory. These notebooks provide an interactive, 
step-by-step guide through the complete research pipeline.

### Launch Jupyter

```bash
# Install Jupyter dependencies
uv sync --extra notebooks

# Start Jupyter notebook server
jupyter notebook examples/
```

### Notebook Workflow

| Notebook | Purpose |
| --- | --- |
| `personas_generation.ipynb` | Generate AI personas using LLM providers (GPT, Claude, Llama) |
| `questionnaires_experiments.ipynb` | Administer personality questionnaires to generated personas |
| `evaluations_table1-3.ipynb` | Reproduce demographic analysis (Tables 1-3 from the paper) |
| `evaluations_table4.ipynb` | Reproduce personality scoring (Table 4) |
| `evaluations_table6.ipynb` | Reproduce reliability analysis (Table 6) |
| `evaluations_table_appendix_A4_A5.ipynb` | Reproduce reliability metrics (Appendix Tables A4-A5) |
| `evaluations_table_appendix_A6_A7.ipynb` | Reproduce accuracy metrics (Appendix Tables A6-A7) |

See [`examples/README.md`](../examples/README.md) for detailed notebook documentation.

## 5. End-to-End Replication Recipe

After restoring the database, verify the research results by running the evaluation notebooks in sequence:

```bash
# Launch Jupyter
jupyter notebook examples/

# Then open and run:
# 1. evaluations_table1-3.ipynb - Demographic analysis
# 2. evaluations_table4.ipynb - Personality scoring
# 3. evaluations_table6.ipynb - Reliability analysis
# 4. evaluations_table_appendix_A4_A5.ipynb - Extended reliability
# 5. evaluations_table_appendix_A6_A7.ipynb - Accuracy metrics
```

Each notebook loads data from the database and reproduces the corresponding analysis from the paper.

## 6. Running New Experiments

To generate new personas and run fresh experiments:

1. **Create an isolated schema** for your experiment:
   ```bash
   export PERSONAS_TARGET_SCHEMA=my_experiment
   make db-upgrade
   ```

2. **Open and run** `examples/personas_generation.ipynb`:
   - Configure models (GPT-4o, Claude, Llama, etc.)
   - Set number of personas to generate
   - Choose experimental conditions (baseline, borderline, etc.)
   - Execute cells to generate personas

3. **Open and run** `examples/questionnaires_experiments.ipynb`:
   - Register experiments for your generated personas
   - Configure questionnaire type (Big Five, EPQR-A)
   - Execute cells to run questionnaires via LLM APIs
   - Results are stored in the database automatically

4. **Analyze results** using the evaluation notebooks:
   - Modify notebook parameters to point to your experimental schema
   - Run analysis cells to generate custom tables and visualizations

## 7. Using the Python API Directly

For advanced use cases, you can import and use the packages programmatically:

```python
from personas_backend.evaluate_questionnaire import (
    register_questionnaire_experiments,
    run_pending_experiments
)

# Register experiments
group_id, exp_ids = register_questionnaire_experiments(
    questionnaire="bigfive",
    model="gpt4o",
    populations=["generated_gpt4o_spain826"],
    schema="personality_trap"
)

# Execute experiments
run_pending_experiments(
    experiments_group_ids=[group_id],
    max_workers=3,
    schema="personality_trap"
)
```

See the notebooks for complete examples.

## 8. Data Exports & Analysis

The evaluation notebooks automatically generate:
- CSV exports of all analysis tables
- Statistical summaries and comparisons
- Visualization charts (where applicable)

For custom analysis, you can also use the `evaluations` package directly:

```python
from evaluations import data_access, table_demographics
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:pass@host/db")
population_df = data_access.load_population(
    conn=engine,
    schema="personality_trap",
    table="personas"
)

# Generate custom demographics table
table_1 = table_demographics.build_demographics_table(
    population_df=population_df,
    models=["GPT-4o"],
    conditions=["Base"]
)
```

## 9. Troubleshooting Checklist

| Issue | Suggested Fix |
| --- | --- |
| Jupyter kernel not found | Run `make jupyter-setup` to configure the kernel |
| Missing credentials errors | Confirm `config.yaml` values or re-export the `PERSONAS_*` environment variables |
| Persona generation fails with API quota errors | Reduce number of personas, rotate providers, or schedule during off-peak hours |
| `psycopg2` import errors | Install PostgreSQL client headers (`sudo apt-get install libpq-dev`) |
| Analysis output is empty | Ensure questionnaire experiments completed successfully and you selected the correct schema |
| Database connection errors | Verify PostgreSQL is running (`make db-status`) and credentials are correct |

For complete dataset documentation, refer to [`../dataset_description.md`](../dataset_description.md).


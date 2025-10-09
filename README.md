# The Personality Trap: LLM Persona Generation Research

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17286313.svg)](https://doi.org/10.5281/zenodo.17286313)

Open-source replication package for **"The Personality Trap: Investigating Demographic Bias in LLM Persona Generation"**, submitted to the ACM Special Issue on Risk and Unintended Harms of Generative AI Systems.

---

## 📊 Dataset Availability

**The research dataset is NOT included in this repository.**

Due to its size (93MB compressed, 698,632 database records), the complete dataset is hosted separately on Zenodo:

**🔗 Download Dataset**: [https://zenodo.org/records/17286313](https://zenodo.org/records/17286313)

The dataset includes:
- PostgreSQL database dump (47MB compressed)
- CSV exports of all tables (280MB)
- 82,600 AI-generated personas across 5 LLMs
- 826 EPQR-A personality profiles
- Complete LLM API request/response logs

**To use this repository**: Download the dataset from Zenodo and follow the [restoration instructions](#step-4-restore-the-research-dataset) below.

---

## ⚠️ Disclaimer

This repository accompanies our research paper and provides the complete replication package for "The Personality Trap" study. 

**About this codebase:**

This is a streamlined, public-facing version of our research team's internal "LLM Personalities Backend" — a comprehensive system we developed to support our ongoing research on demographic bias in large language models. While we have made extensive efforts to simplify and document the code for public use, some complexity remains due to the nature of the research infrastructure.

**Important notes:**

- **Research-grade code**: This codebase was built for rigorous academic research, which sometimes necessitates complexity over simplicity
- **AI-assisted documentation**: We extensively used AI coding agents (GitHub Copilot, Claude) to enhance documentation, add detailed docstrings, and provide clear explanations throughout the codebase
- **Continuous improvement**: We acknowledge there are opportunities for further simplification and improvement
- **Community feedback welcome**: We actively welcome feedback, suggestions, and contributions from the research community

If you encounter unclear documentation, complex workflows, or areas that could be simplified, please:
- Open an issue on GitHub
- Submit a pull request with improvements
- Contact us directly with questions or suggestions

We are committed to making AI bias research accessible and reproducible, and your feedback helps us achieve this goal.

---

## 🔬 Research Context & Dataset

This repository accompanies the paper **"The Personality Trap: Investigating Demographic Bias in LLM Persona Generation"**, submitted to the ACM Special Issue on Risk and Unintended Harms of Generative AI Systems.

### Study Overview

The research investigates whether Large Language Models (LLMs) exhibit demographic bias when generating personas conditioned on personality traits.

**Key findings:**
- **5 LLM models tested**: GPT-3.5, GPT-4o, Claude 3.5 Sonnet, Llama 3.1 70B, Llama 3.2 3B
- **82,600 personas generated** across baseline and borderline personality conditions
- **3 experimental conditions**: Baseline, MaxN (high neuroticism), MaxP (high psychoticism)
- **Demographic bias observed** across all models in gender, race, political orientation, and religious belief

**Complete dataset:**
- **698,632 database records** (9 tables + 1 materialized view)
- **82,600 AI-generated personas** with full demographic profiles
- **541,680 personality questionnaire responses** (EPQR-A and Big Five)
- **All LLM API requests/responses** for reproducibility verification

See [`dataset_description.md`](dataset_description.md) for complete documentation.

---

## 🧪 Extending the Research

This codebase is designed for extensibility. Researchers can:

### Add New Models

Implement a model connector in `src/personas_backend/models_providers/`:

```python
# src/personas_backend/models_providers/my_model_client.py
from personas_backend.models_providers.models_utils import BaseModelClient

class MyModelClient(BaseModelClient):
    def generate_response(self, prompt: str, **kwargs) -> str:
        # Your model API implementation
        pass
```

Register in `models_config.py` and use in experiments via `register_questionnaire_experiments()`.

### Add Custom Questionnaires

Create a questionnaire class in `src/personas_backend/questionnaire/`:

```python
from personas_backend.questionnaire.base_questionnaire import BaseQuestionnaire

class MyQuestionnaire(BaseQuestionnaire):
    def get_questions(self) -> List[Dict]:
        return [{"question": "...", "category": "A", "key": True}, ...]
```

### Add New Demographics

1. Update `Persona` model in `src/personas_backend/db/models.py`
2. Create Alembic migration: `make db-revision msg="Add new field"`
3. Update persona generation prompts and analysis functions

### Create Isolated Experiments

Use schema isolation for parallel experiments:

```bash
export PERSONAS_TARGET_SCHEMA="my_experiment"
make db-upgrade
jupyter notebook examples/personas_generation.ipynb
```

---

## 🤝 Contributing

We welcome contributions! Areas for contribution:

- **New model integrations** (Gemini, local LLMs, fine-tuned models)
- **Additional questionnaires** (NEO-PI-R, HEXACO, Dark Triad)
- **Improved demographic parsing** and classification
- **Performance optimizations** (batch processing, concurrency)
- **Documentation** (tutorials, translations, case studies)

### How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-improvement`
3. Follow existing code style (Black + Ruff + MyPy)
4. Add tests for new functionality
5. Run `make ci` to verify quality checks pass
6. Submit a pull request with clear description

See [`.github/copilot-instructions.md`](.github/copilot-instructions.md) for coding standards.

---

## Citation

If you use this dataset or code in your research, please cite:

**Dataset Citation:**
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

*(Will be updated with final publication details)*

---

## 📝 License

Released under the [MIT License](LICENSE). You are free to:
- ✅ Use for academic research
- ✅ Modify and extend the codebase
- ✅ Use in commercial applications
- ✅ Distribute and sublicense

**Attribution required**: Please cite the paper and link to this repository.

Use the dataset responsibly and follow ethical guidelines when conducting AI bias research.

---

## 👥 Authors & Contact

**Research groupr**: [AId4so](https://blogs.uoc.edu/aid4so/)

For questions, collaboration inquiries, or issues:
- **GitHub Issues**: Report bugs or request features
- **GitHub Discussions**: General questions and research discussions
- **Email**: 

---

## 🙏 Acknowledgments

We thank:
- The open-source community for foundational tools (SQLModel, Pandas, Alembic, Typer)
- Reviewers and editors at ACM for valuable feedback

---

**Ready to replicate the research?** Start with the [Quick Start](#-quick-start-5-steps) guide! 🚀"**, submitted to the ACM Special Issue on Risk and Unintended Harms of Generative AI Systems.

This repository provides everything needed to:
- **Regenerate** 82,600 AI personas using GPT, Claude, and Llama models
- **Replay** 698,632 personality questionnaire responses
- **Reproduce** all evaluation tables and statistical analyses from the paper
- **Extend** the research with new models, populations, or experimental conditions

## 📦 What's Inside

### Core Python Packages

- **`src/personas_backend/`** – Complete persona generation pipeline
  - Database models (SQLModel + Alembic migrations)
  - LLM connectors (OpenAI, AWS Bedrock, local models)
  - Questionnaire administration system
  - Type-safe database access layer
  - CLI tools (Typer) for orchestration

- **`src/evaluations/`** – Research analysis toolkit
  - Demographics tables (Tables 1-3: z-tests on demographic distributions)
  - Personality scores (Table 4: EPQR-A personality dimensions)
  - Reliability analysis (Tables 6, A5: Cronbach's alpha)
  - Accuracy metrics (Tables A6-A7: error rates and completeness)
  - Reusable functions for custom analyses

### Research Artifacts

- **`examples/`** – Jupyter notebooks demonstrating the complete workflow
  - `personas_generation.ipynb` – Generate AI personas with LLMs
  - `questionnaires_experiments.ipynb` – Administer personality questionnaires
  - `evaluations_table*.ipynb` – Reproduce all paper tables (1-6, A4-A7)

- **`dataset/20251008/`** – Complete research dataset
  - Compressed PostgreSQL dump (47MB → 698,632 records)
  - CSV exports for each table
  - Restoration scripts and validation queries

- **`docs/`** – Detailed documentation
  - [`USAGE.md`](docs/USAGE.md) – End-to-end replication guide
  - [`DATABASE_BACKUP.md`](docs/DATABASE_BACKUP.md) – Database restoration instructions

### Development Tools

- **`Makefile`** – One-command shortcuts for common tasks
- **`scripts/`** – Utilities for config validation, schema management, data restoration
- **`tests/`** – Test suite ensuring package reliability


---

## 🚀 Quick Start (5 Steps)

### Step 0: Prerequisites

Before starting, ensure you have:

- **Python 3.12+** ([Download](https://www.python.org/downloads/))
- **Docker & Docker Compose** ([Install Docker](https://docs.docker.com/get-docker/))
- **LLM API keys** for at least one provider:
  - OpenAI API key ([Get key](https://platform.openai.com/api-keys)) – Easiest for testing
  - AWS credentials for Bedrock ([Setup guide](https://docs.aws.amazon.com/bedrock/)) – For Claude/Llama models
  - Local model endpoints (optional) – For self-hosted LLMs

### Step 1: Install Dependencies

Install the project using [uv](https://docs.astral.sh/uv/) (recommended) or pip:

```bash
# Option A: Using uv (recommended - faster, better dependency resolution)
uv sync

# Option B: Using pip
pip install -e .[dev]

# Option C: Using Make (creates .venv automatically)
make setup
```

This installs both Python packages (`personas_backend` and `evaluations`) in editable mode with all dependencies.

### Step 2: Configure Credentials

Create a configuration file from the template:

```bash
cp examples/example_config.yaml config.yaml
```

Edit `config.yaml` to add your credentials, **or** export environment variables:

```bash
# Required: At least one LLM provider
export PERSONAS_OPENAI__API_KEY="sk-your-openai-key"

# Optional: For Claude/Llama via Bedrock
export PERSONAS_BEDROCK__AWS_ACCESS_KEY="your-aws-access-key"
export PERSONAS_BEDROCK__AWS_SECRET_KEY="your-aws-secret-key"
export PERSONAS_BEDROCK__AWS_REGION="us-east-1"

# Required: PostgreSQL connection
export PERSONAS_PG__HOST="localhost"
export PERSONAS_PG__DATABASE="personas"
export PERSONAS_PG__USER="personas_user"
export PERSONAS_PG__PASSWORD="your-password"

# Optional: Isolate your experiments in a custom schema
export PERSONAS_TARGET_SCHEMA="my_experiments"
```

> **Security Note**: Never commit credentials to Git. The `.gitignore` is configured to exclude `.yaml` files (except `.yaml.example`).

### Step 3: Start PostgreSQL Database

Launch a PostgreSQL instance using Docker Compose:

```bash
make db-up          # Starts PostgreSQL 16 in background
make db-status      # Verify connection (should show "Connected")
```

This starts PostgreSQL on `localhost:5432` with credentials from `docker-compose.yml`.

**Alternative**: Use an existing PostgreSQL instance by setting `PERSONAS_PG__*` variables to your server.

### Step 4: Restore the Research Dataset

**Option A: Restore Published Data (Recommended for Replication)**

Restore the complete dataset (698,632 records) from the research backup:

```bash
make db-upgrade     # Apply database schema (Alembic migration)
make db-restore     # Restore data from dataset/20251008/sql/
```

This creates the `personality_trap` schema with all 9 tables and loads:
- 82,600 AI-generated personas
- 19,824 reference personality questionnaires
- 541,680 evaluation responses
- All experiment metadata and configurations

Validation queries and row counts: see [`docs/RESTORATION_VERIFICATION.md`](docs/RESTORATION_VERIFICATION.md).

**Option B: Start Fresh (For New Experiments)**

Create an empty schema and generate your own data:

```bash
make db-upgrade     # Create schema only (no data)
```

Then follow the notebooks in `examples/` to generate personas and run experiments.

### Step 5: Run the Replication Notebooks

Explore the research workflow using Jupyter notebooks:

```bash
# Install Jupyter (if not already installed)
uv sync --extra notebooks

# Launch Jupyter
jupyter notebook examples/
```

**Recommended sequence:**

1. **`personas_generation.ipynb`** – Generate AI personas using LLMs
   - Configure models (GPT-4o, GPT-3.5, Claude, Llama)
   - Generate baseline and borderline personality conditions
   - ~10-30 minutes (2-10 personas per model for testing)

2. **`questionnaires_experiments.ipynb`** – Administer personality questionnaires
   - LLMs impersonate personas and answer questions
   - Stores all API requests/responses for analysis
   - ~15-60 minutes (2 experiments = ~100 questions)

3. **`evaluations_table*.ipynb`** – Reproduce paper tables
   - `evaluations_table1-3.ipynb` – Demographics analysis (Tables 1-3)
   - `evaluations_table4.ipynb` – Personality scores (Table 4)
   - `evaluations_table6.ipynb` – Cronbach's alpha (Table 6)
   - `evaluations_table_appendix_A4_A5.ipynb` – Reliability analysis
   - `evaluations_table_appendix_A6_A7.ipynb` – Accuracy metrics

Each notebook is self-contained with detailed explanations of the research steps.

---

## 📖 Documentation Map

### Getting Started
- **[Quick Start](#-quick-start-5-steps)** (above) – 5-step setup and first run
- **[`examples/QUICKSTART.md`](examples/QUICKSTART.md)** – Minimal 2-persona demo (5 minutes)
- **[`examples/README.md`](examples/README.md)** – Detailed notebook guide

### Database & Setup
- **[`docs/DATABASE_BACKUP.md`](docs/DATABASE_BACKUP.md)** – Backup contents and restoration options
- **[`docs/RESTORATION_VERIFICATION.md`](docs/RESTORATION_VERIFICATION.md)** – Expected row counts and validation queries
- **[`dataset_description.md`](dataset_description.md)** – Complete dataset documentation (all 9 tables + columns)

### Usage & Replication
- **[`docs/USAGE.md`](docs/USAGE.md)** – End-to-end CLI workflow (alternative to notebooks)
- **[`scripts/README.md`](scripts/README.md)** – Utility scripts documentation
- **[`tests/README.md`](tests/README.md)** – Test suite guide

### Development
- **[`Makefile`](Makefile)** – All available commands (run `make help`)
- **[`.github/copilot-instructions.md`](.github/copilot-instructions.md)** – Project architecture and coding standards

---

## 🛠️ Working with the Packages

### `personas_backend` Package

The backend package handles all persona generation and experiment orchestration.

**Key modules:**

- **`db/models.py`** – SQLModel data models (Persona, ExperimentsGroup, EvalQuestionnaires, etc.)
- **`db/schema_config.py`** – Schema resolution helpers (`get_default_schema()`, `get_experimental_schema()`)
- **`db/repositories/`** – Repository pattern for database operations
  - `experiments_repo.py` – Experiment CRUD operations
  - `request_metadata_repo.py` – API metadata persistence

- **`models_providers/`** – LLM integration
  - `openai_client.py` – GPT models (GPT-3.5, GPT-4o)
  - `aws_bedrock_client.py` – Claude and Llama via AWS Bedrock
  - `models_config.py` – Model registry and configuration

- **`evaluate_questionnaire/`** – Questionnaire system
  - `registration.py` – Experiment group and registration API
  - `runner.py` – Parallel questionnaire execution

- **`questionnaire/`** – Questionnaire definitions
  - `bigfive.py` – Big Five personality questionnaire (50 questions)
  - `epqr_a.py` – EPQ-R-A questionnaire (24 questions)

**Example usage:**

```python
from personas_backend.evaluate_questionnaire import (
    register_questionnaire_experiments,
    run_pending_experiments
)

# Register experiments for personas in database
group_id, exp_ids = register_questionnaire_experiments(
    questionnaire="bigfive",
    model="gpt4o",
    populations=["generated_gpt4o_spain826"],
    schema="personality_trap"
)

# Execute experiments (LLM API calls + store results)
run_pending_experiments(
    experiments_group_ids=[group_id],
    max_workers=3,  # Concurrent experiments
    schema="personality_trap"
)
```

### `evaluations` Package

The evaluations package loads results from PostgreSQL and generates analysis tables.

**Key modules:**

- **`data_access.py`** – Data loading utilities
  - `load_population()` – Load personas with demographics
  - `load_questionnaire_experiments()` – Load evaluation results
  - `load_reference_questionnaires()` – Load baseline data

- **`table_demographics.py`** – Demographics analysis (Tables 1-3)
  - Z-tests on demographic proportions
  - Baseline vs borderline comparisons

- **`table_personality.py`** – Personality scores (Table 4)
  - EPQR-A trait calculations (N, E, P, L)
  - Mean ± std by model and condition

- **`table_cronbach.py`** – Reliability analysis (Tables 6, A5)
  - Cronbach's alpha computation
  - Quality thresholds (excellent/good/acceptable/poor)

- **`table_accuracy.py`** – Accuracy metrics (Tables A6-A7)
  - Response rate calculations
  - Error analysis by model

**Example usage:**

```python
from evaluations import data_access, table_demographics
from sqlalchemy import create_engine

# Connect to database
engine = create_engine("postgresql://user:pass@host/db")

# Load data
population_df = data_access.load_population(
    conn=engine,
    schema="personality_trap",
    table="personas"
)

# Generate demographics table (Table 1)
table_1 = table_demographics.build_demographics_table(
    population_df=population_df,
    models=["GPT-4o", "GPT-3.5"],  # Filter to specific models
    conditions=["Base", "MaxN", "MaxP"]  # All conditions
)

print(table_1.to_csv(index=False))
```

Each module exports focused functions for reproducibility and reuse in custom analyses.

---

## 🔧 Makefile Cheat Sheet

The Makefile provides convenient shortcuts for common tasks:

### Setup & Installation
```bash
make setup          # Create .venv + install package
make dev-setup      # Install with dev dependencies
make jupyter-setup  # Configure Jupyter kernel
```

### Database Management
```bash
make db-up          # Start PostgreSQL via Docker Compose
make db-down        # Stop and remove database container
make db-logs        # Tail database logs

make db-upgrade     # Apply latest Alembic migration
make db-status      # Check migration status
make db-revision msg="description"  # Create new migration

make db-restore     # Restore research dataset (SQL backup)
make db-restore-csv # Restore from CSV files (slower, more control)
```

### Code Quality
```bash
make format-all     # Format with Black + Ruff (recommended)
make lint           # Run Ruff linting
make typecheck      # Run MyPy static type checking
make check          # Run lint + typecheck

make test           # Run all tests
make test-fast      # Run fast unit tests only
make test-config    # Run configuration tests
```

### Utilities
```bash
make status         # Show project status (git, db, environment)
make clean          # Remove cache and build files
make notebook       # Start Jupyter notebook server
make config-validate # Validate configuration (OpenAI + DB + Bedrock)
```

### Workflows
```bash
make dev            # Full development setup (setup + db + jupyter)
make ci             # Full CI: format + check + test
make ci-fast        # Faster CI: format + fast tests
```

Run `make help` to see all available targets.

---

## Working with the Packages

### `personas_backend`

- SQLModel data models live in `db/models.py`; schema resolution helpers (`get_default_schema`, `get_experimental_schema`) are centralised in `db/schema_config.py`.
- Repository classes under `db/repositories/` handle persistence for experiment groups, experiment registrations, and request metadata.
- LLM access wrappers (`models_providers/openai_client.py`, `models_providers/aws_bedrock_client.py`) expose a consistent API for persona generation.
- `evaluate_questionnaire/runner.py` replays questionnaires against generated personas in parallel batches, writing results to `eval_questionnaires`.

### `evaluations`

The evaluation package loads results directly from PostgreSQL and returns pandas DataFrames ready for publication:

- `table_demographics.py` – Tables 1-3 (z-tests on demographic distributions).
- `table_personality.py` – Table 4 (EPQR-A personality scores).
- `table_cronbach.py` – Tables 6 and A5 (Cronbach’s alpha).
- `table_accuracy.py` – Tables A6-A7 (accuracy and error metrics).

Each module exposes small focused functions so you can script custom analyses or reuse them in notebooks.

---

## Makefile Cheatsheet

- `make setup` – create `.venv` and install the package.
- `make format` / `make format-all` – Black + Ruff formatting.
- `make lint`, `make typecheck`, `make test`, `make test-fast` – quality and test suites.
- `make db-up`, `make db-down`, `make db-logs` – manage the Dockerised Postgres instance.
- `make db-upgrade`, `make db-status`, `make db-revision msg="..."` – Alembic migrations.
- `make db-restore`, `make db-restore-csv` – restore the published dataset (SQL or CSV path).
- `make status` – consolidated project status report.

---

---

## 🔬 Research Context & Dataset

---

## Dataset Overview

- **82,600** personas generated across five LLMs (GPT-3.5, GPT-4o, Claude 3.5 Sonnet, Llama 3.1 70B, Llama 3.2 3B).
- **826** baseline personality profiles (Big Five) used to seed the study.
- **700,623** total records across 9 tables plus the `experiments_evals` materialized view.
- **3** experimental conditions: Baseline, MaxN (borderline neuroticism), MaxP (borderline psychoticism).
- Synthetic personas only — no human subject data or identifying information.

Refer to [`dataset_description.md`](dataset_description.md) for column-level details and provenance.

---

## Testing & Verification

- `make test` runs the pytest suite (no external API calls).
- `make status` checks Docker, migrations, configuration, and outstanding Alembic revisions.
- After restoring the dataset, you can verify everything works by running the evaluation notebooks in `examples/`:
  - `evaluations_table1-3.ipynb` – Verify demographic analysis loads correctly
  - `evaluations_table4.ipynb` – Verify personality scoring calculations
  - `questionnaires_experiments.ipynb` – Test questionnaire administration (requires API keys)

---

## Citation

**Dataset Citation:**
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

---

## License

Released under the [MIT License](LICENSE). Please use the dataset responsibly and follow ethical guidelines when conducting AI bias research.

# The Personality Trap: LLM Persona Generation Research

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17286313.svg)](https://doi.org/10.5281/zenodo.17286313)

Open-source replication package for **"The Personality Trap: Investigating Demographic Bias in LLM Persona Generation"**, submitted to the ACM Special Issue on Risk and Unintended Harms of Generative AI Systems.

---

## 📊 Dataset Availability

**The research dataset is NOT included in this repository.**
Download from Zenodo: **https://zenodo.org/records/17286313**

Contents (external):
- PostgreSQL dump and CSV exports
- 82,600 AI-generated personas across 5 LLMs
- 826 personality profiles and 541,680 questionnaire responses
- Full API request/response logs

Follow the restore steps below after downloading.

---

## ⚠️ Disclaimer

This is a research codebase originally built for internal experiments. Complexity reflects research needs; contributions to simplify are welcome. AI-assisted documentation was used throughout.

---

## 🔬 Research Context

- **Goal:** Measure demographic bias in LLM persona generation.
- **Models:** GPT-3.5, GPT-4o, Claude 3.5 Sonnet, Llama 3.1 70B, Llama 3.2 3B.
- **Conditions:** Baseline, MaxN (neuroticism), MaxP (psychoticism).
- **Scale:** 82,600 personas, 541,680 questionnaire responses, 9 tables + 1 materialized view.

See `dataset_description.md` and the publications in `publications/`.

---

## 📦 What’s Inside

- `src/personas_backend/` — Persona generation, questionnaire execution, DB access, model clients.
- `src/evaluations/` — Analysis helpers for demographics, personality scores, reliability, and accuracy.
- `examples/` — Notebooks for generation, questionnaires, and reproducing tables.
- `docs/` — Usage guides, backup/restore instructions, validation queries.
- `tools/pipeline.py` — Typer CLI covering end-to-end workflows.

---

## 🚀 Quick Start (5 Steps)

1) **Install dependencies**
```bash
uv sync               # or: pip install -e .[dev]
```
2) **Create config**
```bash
cp .yaml.example .yaml
# fill in OpenAI/Bedrock keys and Postgres credentials
```
3) **Start PostgreSQL**
```bash
make db-up
make db-status
```
4) **Apply schema & restore data (optional)**
```bash
make db-upgrade
# after downloading Zenodo dataset:
make db-restore      # or follow docs/RESTORATION_VERIFICATION.md
```
5) **Run a quick pipeline**
```bash
uv run python tools/pipeline.py run-evals --questionnaire epqra --model gpt4o
```

---

## 📖 Documentation Map

- Quick demo: `examples/QUICKSTART.md`
- Detailed usage: `docs/USAGE.md`
- Database restore: `docs/DATABASE_BACKUP.md`, `docs/RESTORATION_VERIFICATION.md`
- Dataset description: `dataset_description.md`
- Scripts reference: `scripts/README.md`
- Tests overview: `tests/README.md`
- Validation scripts: `docs/VALIDATION.md`
- Checklists: `docs/checklists/PRE_RELEASE_CHECKLIST.md` and `docs/REPRODUCTION_CHECKLIST.md`
- Reproduction steps: `docs/REPRODUCTION_CHECKLIST.md`
- Ethics & security: `ETHICS.md`, `SECURITY.md`

---

## 🛠️ Troubleshooting

- **Database connection refused**
  - Check Docker: `docker ps | grep postgres`
  - Verify port: `lsof -i :5432`
  - Validate config: `make config-validate`
- **Missing API keys**
  - Set env vars (e.g., `export PERSONAS_OPENAI__API_KEY="sk-..."`) or update `.yaml`.
- **Module not found (`personas_backend`)**
  - Install in editable mode: `pip install -e .` (or `uv sync`).
- **Schema/table missing**
  - Run migrations: `make db-upgrade`
  - Restore dataset if needed: `make db-restore`
- **Tests failing**
  - Ensure DB is up and config is valid: `make db-status`, `make test-config`
  - Run fast suite: `make test-fast`

---

## 🧪 Working with the Packages

**Persona generation (Python)**
```python
from personas_backend.persona_generator import PersonaGenerator
from personas_backend.core.enums import ModelID

gen = PersonaGenerator()
# supply questionnaire dataframe, model, and optional personality_id
result = gen.generate_single_persona(questionnaire_df=df, model=ModelID.GPT4O)
```

**Questionnaire registration + execution (Python)**
```python
from personas_backend.evaluate_questionnaire import (
    register_questionnaire_experiments,
    run_pending_experiments,
)

group_id, exp_ids = register_questionnaire_experiments(
    questionnaire="bigfive",
    model="gpt4o",
    populations=["generated_gpt4o_spain826"],
)
run_pending_experiments([group_id], max_workers=3)
```

**CLI (Typer)**
```bash
# Generate personas (DB-backed)
uv run python tools/pipeline.py generate-personas --run-id demo --model gpt4o
# Register and run evaluations
uv run python tools/pipeline.py run-evals --questionnaire epqra --model gpt4o
```

---

## 🔧 Makefile Cheat Sheet

- `make setup` — create `.venv` and install
- `make format-all` — Black + Ruff
- `make lint` / `make typecheck` / `make test`
- `make db-up` / `make db-down` / `make db-status`
- `make db-upgrade` / `make db-revision msg="..."` / `make db-restore`
- `make config-validate` — check OpenAI/Bedrock/DB config

Run `make help` to see all targets.

---

## 🤝 Contributing

Contributions welcome: new model integrations, questionnaires, demographics parsing, performance, documentation, and tests. Please run `make ci` before opening a PR. See `CONTRIBUTING.md` (forthcoming) for details.

---

## 📚 Citation

**Dataset**
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

**Paper**
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

## 📝 License

MIT License. See `LICENSE`.

---

## 👥 Authors & Contact

**Principal Investigator:** Gregorio Ferreira
Contact: [Your contact email for research inquiries]

---

## 🙏 Acknowledgments

Thanks to the open-source community (SQLModel, Pandas, Alembic, Typer) and reviewers for feedback. Community contributions are encouraged.

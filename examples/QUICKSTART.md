# Quick Start Guide - The Personality Trap Research Replication

Get started replicating the research in 5 minutes!

## Prerequisites Checklist

- [ ] Python 3.12+ installed
- [ ] PostgreSQL database running (Docker or local)
- [ ] At least one LLM API key configured (OpenAI recommended for testing)

## Setup (One-time)

### 1. Clone and Install
```bash
cd /path/to/the_personality_trap
uv sync  # or: pip install -e .
```

### 2. Start Database
```bash
make db-up  # Starts PostgreSQL via Docker Compose
```

### 3. Configure API Keys
Create or edit your config file:
```bash
cp examples/example_config.yaml my_config.yaml
```

Edit `my_config.yaml`:
```yaml
openai:
  api_key: "sk-your-key-here"  # Get from platform.openai.com

schema:
  default_schema: "personality_trap"  # Production (read-only)
  target_schema: "quick_test"         # Your experimental schema
```

**OR** use environment variables:
```bash
export OPENAI_API_KEY="sk-your-key-here"
export PERSONAS_TARGET_SCHEMA="quick_test"
```

### 4. Verify Setup
```bash
make status  # Should show "Connected to database"
```

## Run the Notebooks

### Phase 1: Generate Personas (10 minutes)

Open `examples/personas_generation.ipynb` in Jupyter:

```bash
jupyter notebook examples/personas_generation.ipynb
```

**Recommended settings for testing:**
```python
N_PERSONALITIES = 2           # Just 2 personalities
baseline_models = [ModelID.GPT4O]  # Just GPT-4o
GENERATE_BORDERLINE = False   # Skip borderline personas
```

**Run all cells** (Cell → Run All)

**Expected output:**
- ✅ Schema created
- ✅ 2 reference personalities loaded
- ✅ 2 personas generated
- ✅ Data verified

### Phase 2: Run Experiments (15 minutes)

Open `examples/questionnaires_experiments.ipynb`:

```bash
jupyter notebook examples/questionnaires_experiments.ipynb
```

**Recommended settings for testing:**
```python
QUESTIONNAIRE_TYPE = "bigfive"
EXPERIMENT_MODEL = ModelID.GPT4O
REPETITIONS = 1
POPULATIONS_TO_TEST = ["generated_gpt4o_spain826"]
```

**Run all cells** (Cell → Run All)

**Expected output:**
- ✅ 2 personas found
- ✅ 2 experiments registered
- ✅ 2 experiments executed
- ✅ 100 questionnaire answers collected (2 personas × 50 questions)

## Verify Results

### Check Database
```bash
# Connect to database
psql -h localhost -U postgres -d personality_trap_db

# Check personas
SELECT COUNT(*) FROM quick_test.personas;
-- Expected: 2

# Check experiments
SELECT COUNT(*) FROM quick_test.experiments_list;
-- Expected: 2

# Check answers
SELECT COUNT(*) FROM quick_test.eval_questionnaires;
-- Expected: 100 (2 personas × 50 questions)
```

### View Results in Notebook
The last cell of each notebook displays summary statistics and sample data.

## Next Steps

### For Testing/Learning
You're done! You've successfully:
- ✅ Created an experimental schema
- ✅ Generated AI personas using GPT-4o
- ✅ Collected personality questionnaire responses
- ✅ Stored all data in PostgreSQL

### For Full Replication

**Increase scale in `personas_generation.ipynb`:**
```python
N_PERSONALITIES = 50  # More reference personalities
baseline_models = [
    ModelID.GPT35,
    ModelID.GPT4O,
    ModelID.CLAUDE35_SONNET,
    ModelID.LLAMA3_23B,
    ModelID.LLAMA3_170B,
]
GENERATE_BORDERLINE = True  # Include experimental variants
```

**Run multiple experiments in `questionnaires_experiments.ipynb`:**
```python
REPETITIONS = 3  # Statistical validity
# Run for both questionnaire types:
# - bigfive
# - epqr_a
```

**Expected processing time:**
- 50 personalities × 5 models = 250 personas (~2-3 hours)
- 250 personas × 2 questionnaires × 3 reps = 1500 experiments (~6-8 hours)

## Cost Estimates (OpenAI GPT-4o)

**Testing (2 personas):**
- Persona generation: ~$0.10
- Questionnaire experiments: ~$0.20
- **Total: ~$0.30**

**Full replication (50 personalities × 5 models):**
- Persona generation: ~$12-15
- Questionnaire experiments: ~$30-40
- **Total: ~$50-60** (varies by model mix)

*Note: Claude and Llama have different pricing. Check your provider.*

## Troubleshooting

### "Schema does not exist"
**Fix:** Run the schema creation cell in `personas_generation.ipynb` (cell 3)

### "No personas found"
**Fix:** Complete `personas_generation.ipynb` before running `questionnaires_experiments.ipynb`

### "API key not found"
**Fix:**
```bash
export OPENAI_API_KEY="sk-your-key-here"
# Then restart Jupyter kernel
```

### "Rate limit exceeded"
**Fix:** Reduce concurrent workers:
```python
max_workers = 1  # Instead of 3
```

### Database connection failed
**Fix:**
```bash
make db-up  # Ensure PostgreSQL is running
make status # Verify connection
```

## Common Questions

**Q: Can I use a different database?**
A: Yes, but you'll need to modify connection settings in your config file.

**Q: Do I need all 5 LLM providers?**
A: No! Start with just OpenAI (GPT-4o). Add others as needed.

**Q: Can I run this without Docker?**
A: Yes, configure PostgreSQL connection to your existing database instance.

**Q: How do I reset and start over?**
A: Drop your experimental schema:
```sql
DROP SCHEMA quick_test CASCADE;
```
Then re-run the notebooks.

**Q: Where is the data stored?**
A: PostgreSQL database in schema `quick_test` (or whatever you configured as `target_schema`)

**Q: Can I export the data?**
A: Yes! Both notebooks include export examples:
```python
personas_df.to_csv('personas.csv', index=False)
answers_df.to_csv('answers.csv', index=False)
```

## Support

📚 **Full documentation:** `examples/README.md`
🗄️ **Database schema:** `examples/DATABASE_SCHEMA.md`
🔧 **Updates summary:** `examples/UPDATES_SUMMARY.md`

## Success Criteria

You know it's working when:
- ✅ Both notebooks run without errors
- ✅ Database tables are populated
- ✅ Verification cells show expected counts
- ✅ Sample data displays correctly

Happy researching! 🎉

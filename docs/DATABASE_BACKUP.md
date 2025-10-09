# Database Restoration Guide

This guide explains how to restore the research dataset that powers **The Personality Trap** replication package. The dataset
contains all personas, questionnaires, experiment metadata, and derived evaluation tables referenced in the paper.

## Quick Start

**For new users, use the automated restore script:**

```bash
./dataset/20251008/restore_backup.sh
```

**What this does:**
- ✅ Starts PostgreSQL container (Docker Compose)
- ✅ Creates schema with Alembic migrations
- ✅ Restores all 698,632 records
- ✅ Verifies data integrity
- ✅ Tests auto-increment functionality

**Time required:** ~2-3 minutes

See [Automated Restoration](#automated-restoration-with-script) for details.

---

## Dataset Contents

The compressed backup archive `personas_database_backup.sql.gz` (47MB) captures the full PostgreSQL schema and data required to
reproduce the published results. The dataset includes:

- **9 tables** with **698,632 total records** in the `personality_trap` schema
- 82,600 AI-generated personas with demographic annotations
- 19,824 reference personality questionnaires (baseline population)
- 19,824 random questionnaires (control baseline)
- 541,680 evaluated questionnaire responses
- 18,573 experiment request/response metadata records
- 17,346 experiment registrations
- 752 master questionnaire definitions
- 22 experiment group configurations

**Location:** `dataset/20251008/sql/personas_database_backup.sql.gz`

**Additional formats:**
- Schema-only DDL: `dataset/20251008/sql/schema_export.sql` (28KB)
- CSV exports: `dataset/20251008/csv/` (280MB total, all 9 tables)
- View creation: `dataset/20251008/sql/create_experiments_evals_view.sql`

For complete dataset documentation, see [`dataset_description.md`](../dataset_description.md).

---

## Prerequisites

### Option 1: Using Docker (Recommended)

- Docker and Docker Compose installed
- Repository cloned locally
- 500MB free disk space

**No PostgreSQL installation required** - the Docker Compose file provides PostgreSQL 16 automatically.

### Option 2: Manual Restoration

1. **PostgreSQL 16** (or compatible version 14+) available locally or remotely
2. **Database user** with privileges to create schemas, tables, and roles in the target database
3. Environment variables matching the [configuration guide](../README.md#step-2-configure-credentials):
   - `PERSONAS_PG__HOST`
   - `PERSONAS_PG__PORT` (defaults to 5432 if omitted)
   - `PERSONAS_PG__DATABASE`
   - `PERSONAS_PG__USER`
   - `PERSONAS_PG__PASSWORD`
4. Optional: `PERSONAS_TARGET_SCHEMA` to restore into a non-default schema. When unset, the backup restores into the
   `personality_trap` schema

> **Security reminder:** never commit credentials to Git. Prefer `.env` files excluded by `.gitignore` or export the variables
> in your shell session before running the restoration script.

---

## Restoration Methods

### Automated Restoration with Script

**Best for:** First-time setup, clean rebuilds, CI/CD automation

```bash
# From repository root
./dataset/20251008/restore_backup.sh
```

**What the script does:**

1. **Prerequisites Check** - Verifies Docker, backup files, directory structure
2. **Database Startup** - Starts PostgreSQL container via `docker compose up -d postgres`
3. **Schema Creation** - Applies Alembic migrations (`make db-upgrade`)
   - Creates `personality_trap` schema
   - Creates all 9 tables with indexes and constraints
   - Sets up auto-increment sequences
   - Configures server-side timestamp defaults
4. **Data Restoration** - Decompresses and loads `personas_database_backup.sql.gz`
   - Restores all 698,632 records
   - Filters expected "already exists" errors (indexes from migration)
5. **Cleanup** - Removes old Alembic version entries, ensures clean migration state
6. **Verification** - Counts rows in all tables, checks migration status
7. **Auto-Increment Test** - Inserts/removes test record to verify functionality

**Example output:**
```
[INFO] Database Restore Script
[INFO] Backup: dataset/20251008
[STEP] Checking prerequisites...
[INFO] Prerequisites check passed ✓
[STEP] Starting database container...
[INFO] Database is ready ✓
[STEP] Applying database migrations...
[INFO] Migrations applied ✓
[STEP] Restoring SQL backup...
[INFO] SQL backup restored ✓
[STEP] Verifying restoration...
personas: 82600 rows
experiment_request_metadata: 18575 rows
... (all tables)
[INFO] Migration status: 001_initial_schema (head)
[STEP] Testing auto-increment functionality...
[INFO] Auto-increment test passed ✓
[INFO] Restore completed successfully! 🎉
```

### Using Makefile Commands

**Best for:** Development workflow integration

```bash
# Start database
make db-up

# Create schema
make db-upgrade

# Restore data
make db-restore

# Verify
make db-status
```

Each Make target provides focused functionality and can be run independently.

### Manual Restoration with psql

**Best for:** Custom deployment pipelines, remote databases, non-Docker environments

```bash
# Option 1: Direct restore using local psql (requires PostgreSQL 14+ client)
PGPASSWORD="$PERSONAS_PG__PASSWORD" gunzip -c dataset/20251008/sql/personas_database_backup.sql.gz | \
  psql -h "$PERSONAS_PG__HOST" -p "${PERSONAS_PG__PORT:-5432}" -U "$PERSONAS_PG__USER" -d "$PERSONAS_PG__DATABASE"

# Option 2: Using Docker for version compatibility
docker run --rm --network host \
  -e PGPASSWORD="$PERSONAS_PG__PASSWORD" \
  -v "$(pwd)/dataset/20251008/sql:/backup" \
  postgres:16 \
  bash -c "gunzip -c /backup/personas_database_backup.sql.gz | psql -h $PERSONAS_PG__HOST -U $PERSONAS_PG__USER -d $PERSONAS_PG__DATABASE"

# Option 3: Decompress first, then restore
gunzip -c dataset/20251008/sql/personas_database_backup.sql.gz > /tmp/personas_database.sql
PGPASSWORD="$PERSONAS_PG__PASSWORD" psql \
  -h "$PERSONAS_PG__HOST" -p "${PERSONAS_PG__PORT:-5432}" \
  -U "$PERSONAS_PG__USER" -d "$PERSONAS_PG__DATABASE" \
  -f /tmp/personas_database.sql
rm /tmp/personas_database.sql  # Clean up temporary file
```

The backup includes `--clean --if-exists` statements, so it will safely drop and recreate all objects.

---

## Creating New Backups

**To create a new backup of your current database state:**

```bash
./dataset/20251008/create_backup.sh
```

**What this creates:**

1. **Compressed SQL dump** - `personas_database_backup.sql.gz` (47MB)
   - Full schema and data backup
   - Includes `--clean --if-exists` for safe restoration

2. **Schema-only export** - `schema_export.sql` (28KB)
   - DDL only (CREATE TABLE, indexes, constraints)
   - Useful for schema documentation

3. **CSV exports** - `csv/*.csv` (280MB total)
   - All 9 tables exported with headers
   - Human-readable format for external analysis tools

4. **Materialized view script** - `create_experiments_evals_view.sql`
   - Script to recreate the `experiments_evals` view

5. **Documentation** - `table_counts.txt`, updated `README.md`
   - Row counts for verification
   - Auto-generated documentation

**Use when:**
- Preserving current state before major changes
- Creating snapshots for versioning
- Archiving research data
- Sharing datasets with collaborators

**Prerequisites:**
- Docker container running (`make db-up`)
- Database populated with data

**Output structure:**
```
dataset/20251008/
├── sql/
│   ├── personas_database_backup.sql.gz   # 47MB compressed
│   ├── schema_export.sql                 # 28KB DDL
│   └── create_experiments_evals_view.sql # 9KB view script
├── csv/
│   ├── personas.csv                      # 136MB
│   ├── experiment_request_metadata.csv   # 123MB
│   └── ... (7 more CSV files)
└── table_counts.txt                      # Row counts
```

---

## Verifying the Restore

After the restore completes, validate the dataset by checking table counts:

### Using Make Commands

```bash
# Check migration status
make db-status
# Expected: 001_initial_schema (head)

# Get database statistics
docker exec personas_postgres psql -U personas -d personas -c \
    "SELECT schemaname, tablename, n_live_tup FROM pg_stat_user_tables WHERE schemaname = 'personality_trap' ORDER BY tablename;"
```

### Using psql Directly

```bash
# Check table counts
PGPASSWORD="$PERSONAS_PG__PASSWORD" psql \
  -h "$PERSONAS_PG__HOST" -p "${PERSONAS_PG__PORT:-5432}" \
  -U "$PERSONAS_PG__USER" -d "$PERSONAS_PG__DATABASE" \
  -c "SELECT schemaname, tablename, n_live_tup FROM pg_stat_user_tables WHERE schemaname = 'personality_trap' ORDER BY tablename;"

# Expected output (approximate row counts):
# personality_trap | alembic_version              | 1
# personality_trap | eval_questionnaires          | 541,680
# personality_trap | experiment_request_metadata  | 18,573
# personality_trap | experiments_groups           | 22
# personality_trap | experiments_list             | 17,346
# personality_trap | personas                     | 82,600
# personality_trap | questionnaire                | 752
# personality_trap | random_questionnaires        | 19,824
# personality_trap | reference_questionnaires     | 19,824
```

### Verify Materialized View and Indexes

```bash
# Check materialized view
PGPASSWORD="$PERSONAS_PG__PASSWORD" psql \
  -h "$PERSONAS_PG__HOST" -U "$PERSONAS_PG__USER" -d "$PERSONAS_PG__DATABASE" \
  -c "SELECT COUNT(*) FROM personality_trap.experiments_evals;"

# List all indexes
PGPASSWORD="$PERSONAS_PG__PASSWORD" psql \
  -h "$PERSONAS_PG__HOST" -U "$PERSONAS_PG__USER" -d "$PERSONAS_PG__DATABASE" \
  -c "SELECT schemaname, tablename, indexname FROM pg_indexes WHERE schemaname = 'personality_trap' ORDER BY tablename, indexname;"
```

### Test Auto-Increment (Optional)

```bash
# Test that auto-increment sequences work correctly
docker exec personas_postgres psql -U personas -d personas -c \
    "INSERT INTO personality_trap.experiment_request_metadata (experiment_id) \
     VALUES (1) RETURNING id, created;"
# Should return next sequential ID with server-generated timestamp

# Clean up test data
docker exec personas_postgres psql -U personas -d personas -c \
    "DELETE FROM personality_trap.experiment_request_metadata WHERE experiment_id = 1;"
```

For comprehensive verification, see [`RESTORATION_VERIFICATION.md`](./RESTORATION_VERIFICATION.md).

---

## Troubleshooting

| Symptom | Resolution |
| --- | --- |
| `psql: error: connection to server failed` | Confirm `PERSONAS_PG__HOST`, network access, and firewall rules. Check PostgreSQL is running (`make db-up`). |
| `permission denied for schema` | Ensure the database user has `CREATE` privileges: `GRANT CREATE ON DATABASE personas TO your_user;` |
| `gunzip: command not found` | Install gzip utility (`sudo apt-get install gzip` on Debian/Ubuntu, `brew install gzip` on macOS). |
| `version mismatch` errors | Use Docker with `postgres:16` image as shown in the restoration examples. |
| Restore succeeds but tables are empty | Verify you're checking the correct schema: `\dn` in psql to list schemas. |
| `schema "personality_trap" does not exist` | The backup creates the schema automatically; check for errors in restore output. If using manual restore, run migrations first: `make db-upgrade` |
| Very slow restore | Normal for 698K records. Full restore takes ~1-5 minutes depending on hardware. |
| "already exists" errors during restore | Expected when using the automated script (indexes created by Alembic migration). These are filtered automatically. |
| Container not starting | Check Docker daemon: `docker ps`. Ensure ports 5432/5433 are not in use: `lsof -i :5432` |
| Backup file not found | Ensure you're running from repository root directory. Check path: `ls -lh dataset/20251008/sql/personas_database_backup.sql.gz` |

---

## Next Steps After Restore

1. **Verify installation:**
   ```bash
   make status
   ```

2. **Explore the data:**
   ```bash
   # Connect to database
   docker exec -it personas_postgres psql -U personas -d personas
   
   # List all tables
   \dt personality_trap.*
   
   # Sample personas data
   SELECT id, age, gender, ethnicity, nationality FROM personality_trap.personas LIMIT 10;
   ```

3. **Run analysis notebooks:**
   ```bash
   cd examples
   jupyter notebook evaluations_table1-3.ipynb
   ```

4. **Generate new personas (optional):**
   ```bash
   uv run python tools/pipeline.py generate-personas --model gpt4o --count 100
   ```

For full usage documentation, see [`USAGE.md`](./USAGE.md).

---

## Additional Resources

- **[`dataset_description.md`](../dataset_description.md)** - Complete dataset reference with all column descriptions
- **[`RESTORATION_VERIFICATION.md`](./RESTORATION_VERIFICATION.md)** - Detailed verification checklist
- **[`examples/DATABASE_SCHEMA.md`](../examples/DATABASE_SCHEMA.md)** - Schema reference for notebook users
- **[Main README](../README.md)** - Quick start guide and project overview

## Dataset Statistics

After successful restore, you'll have access to:

- **Total Records**: 698,632 across 9 tables
- **Uncompressed Size**: ~500MB
- **Compressed Size**: 47MB (gzip)
- **PostgreSQL Version**: Created with 16.10, compatible with 14+
- **Schema**: `personality_trap`
- **Tables**: 9 (see dataset_description.md for details)
- **Indexes**: 33 across all tables
- **Materialized Views**: 1 (`experiments_evals`)

With the restored database in place you can immediately execute the replication pipeline described in
[USAGE.md](USAGE.md) and explore the data using the notebooks in `examples/`.

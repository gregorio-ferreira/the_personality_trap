# Documentation

This directory contains detailed documentation for **The Personality Trap** research replication package.

## 📚 Available Guides

### For Users

**[`USAGE.md`](USAGE.md)** - Complete CLI-based workflow guide
- End-to-end replication using command-line tools
- Alternative to Jupyter notebooks for automated workflows
- Pipeline CLI commands and orchestration
- **Best for:** Researchers who prefer scripts over notebooks

**[`DATABASE_BACKUP.md`](DATABASE_BACKUP.md)** - Database restoration guide
- Detailed backup restoration instructions
- Multiple restoration methods (Docker, local psql, manual)
- Troubleshooting common issues
- Verification queries
- **Best for:** Setting up the research database

**[`RESTORATION_VERIFICATION.md`](RESTORATION_VERIFICATION.md)** - Verification checklist
- Expected record counts for all tables
- Data integrity validation queries
- Post-restoration checks
- **Best for:** Confirming successful database restoration

### For Developers

**[`ARCHITECTURE.md`](ARCHITECTURE.md)** - Technical architecture overview
- Experiments registration and execution system
- Database integration patterns and repositories
- CLI orchestration (Typer-based pipeline)
- Testing approach and troubleshooting
- **Best for:** Understanding the codebase internals or extending functionality

**[`MULTI_SCHEMA_ALEMBIC.md`](MULTI_SCHEMA_ALEMBIC.md)** - Multi-schema database migrations
- Using Alembic with multiple PostgreSQL schemas
- Independent version tracking per schema
- Testing and production isolation strategies
- Configuration and usage examples
- **Best for:** Working with multiple schemas or understanding migration architecture

## 🗂️ Other Documentation

Documentation is distributed across the repository for easy access:

**Root directory:**
- [`../README.md`](../README.md) - Main entry point (start here!)
- [`../dataset_description.md`](../dataset_description.md) - Complete dataset reference

**Examples directory:**
- [`../examples/README.md`](../examples/README.md) - Notebook-based workflow guide
- [`../examples/QUICKSTART.md`](../examples/QUICKSTART.md) - 5-minute quick start
- [`../examples/DATABASE_SCHEMA.md`](../examples/DATABASE_SCHEMA.md) - Schema reference for notebooks

**Utilities:**
- [`../scripts/README.md`](../scripts/README.md) - Utility scripts documentation
- [`../tests/README.md`](../tests/README.md) - Test suite guide

## 🚀 Getting Started

**New to the project?** Start with the [main README](../README.md) which provides:
- 5-step quick start guide
- Package overview
- Installation instructions
- Links to all documentation

**Already familiar?** Jump to:
- **Notebooks workflow** → [`../examples/README.md`](../examples/README.md)
- **CLI workflow** → [`USAGE.md`](USAGE.md)
- **Database setup** → [`DATABASE_BACKUP.md`](DATABASE_BACKUP.md)
- **Dataset reference** → [`../dataset_description.md`](../dataset_description.md)

## 💡 Documentation Tips

1. **Start simple**: Begin with the main README quick start
2. **Choose your path**: Notebooks (interactive) or CLI (automated)
3. **Reference as needed**: Deep-dive docs are here when you need them
4. **Ask questions**: Open GitHub Issues for clarification

---

**Questions?** See the main [README](../README.md) for contact information and community resources.

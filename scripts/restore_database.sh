#!/bin/bash
# Script to restore the personas database from backup
# This script will:
# 1. Start the PostgreSQL container if not running
# 2. Restore the complete database from backup
# 3. Verify the restore was successful

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting database restore process...${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Check if backup file exists
BACKUP_FILE="personas_database_backup.sql.gz"
if [ ! -f "$BACKUP_FILE" ]; then
    echo -e "${RED}Error: Backup file $BACKUP_FILE not found.${NC}"
    echo "Please ensure the backup file is in the current directory."
    exit 1
fi

echo -e "${YELLOW}Starting PostgreSQL container...${NC}"
# Start the PostgreSQL container using docker-compose
make db-up

echo -e "${YELLOW}Waiting for PostgreSQL to be ready...${NC}"
# Wait for PostgreSQL to be ready
sleep 10

# Check if container is running
if ! docker ps | grep -q personas_postgres; then
    echo -e "${RED}Error: PostgreSQL container is not running.${NC}"
    exit 1
fi

echo -e "${YELLOW}Restoring database from backup...${NC}"
# Decompress and restore the database
gunzip -c "$BACKUP_FILE" | docker exec -i personas_postgres psql -U personas

echo -e "${YELLOW}Verifying restore...${NC}"
# Verify the restore was successful by checking table counts
EXPECTED_TABLES=6
ACTUAL_TABLES=$(docker exec personas_postgres psql -U personas -d personas -t -c "
    SELECT COUNT(*)
    FROM information_schema.tables
    WHERE table_schema = 'personality_trap'
")

if [ "$ACTUAL_TABLES" -eq "$EXPECTED_TABLES" ]; then
    echo -e "${GREEN}✅ Database restore successful!${NC}"
    echo
    echo "Tables in personality_trap schema:"
    docker exec personas_postgres psql -U personas -d personas -c "
        SELECT
            table_name,
            (SELECT COUNT(*) FROM personality_trap.personas) as personas_count,
            (SELECT COUNT(*) FROM personality_trap.experiments_groups) as groups_count,
            (SELECT COUNT(*) FROM personality_trap.experiments_list) as experiments_count,
            (SELECT COUNT(*) FROM personality_trap.eval_questionnaires) as evaluations_count,
            (SELECT COUNT(*) FROM personality_trap.reference_questionnaires) as ref_questions_count,
            (SELECT COUNT(*) FROM personality_trap.questionnaire) as questions_count
        FROM information_schema.tables
        WHERE table_schema = 'personality_trap'
        LIMIT 1
    "
else
    echo -e "${RED}❌ Database restore may have failed. Expected $EXPECTED_TABLES tables, found $ACTUAL_TABLES.${NC}"
    exit 1
fi

echo
echo -e "${GREEN}Database is ready for use!${NC}"
echo "You can now run experiments and analysis using the restored data."
echo
echo "To run migrations (if needed): make db-upgrade"
echo "To run validation tests: make check"

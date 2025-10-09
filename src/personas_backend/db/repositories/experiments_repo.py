from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Optional

from personas_backend.db.models import (
    ExperimentsGroup,
    ExperimentsList,
)
from sqlmodel import Session, select


class ExperimentsRepository:
    """Repository for experiment and group CRUD using SQLModel sessions.

    **Schema Handling:**
    SQLModel models have static `__table_args__` defined at import time which
    defaults to the production schema. To write to experimental schemas,
    this repository uses raw SQL for inserts to explicitly qualify the schema.
    """

    def __init__(self, session: Session, schema: str | None = None):
        """Initialize repository.

        Args:
            session: SQLModel session
            schema: Target database schema. If None, uses experimental schema
                    from ConfigManager (schema.target_schema).
        """
        self.session = session
        # Determine target schema
        if schema is None:
            from personas_backend.db.schema_config import get_experimental_schema

            schema = get_experimental_schema()
        self.schema = schema

    # --- Experiment Groups ---
    def create_group(
        self,
        description: str,
        system_role: str,
        base_prompt: str,
        translated: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        ii_repeated: bool = False,
    ) -> ExperimentsGroup:
        """Create experiment group in the configured schema using raw SQL with auto-increment ID."""
        from datetime import datetime

        from sqlalchemy import text

        # Insert with explicit schema, letting PostgreSQL sequence auto-generate ID
        # Use RETURNING to get the auto-generated experiments_group_id
        insert_query = text(
            f"""
            INSERT INTO {self.schema}.experiments_groups (
                created,
                description,
                system_role,
                base_prompt,
                translated,
                temperature,
                top_p,
                ii_repeated,
                concluded,
                processed
            ) VALUES (
                :created,
                :description,
                :system_role,
                :base_prompt,
                :translated,
                :temperature,
                :top_p,
                :ii_repeated,
                FALSE,
                FALSE
            )
            RETURNING experiments_group_id
        """
        )

        result = self.session.execute(
            insert_query,
            {
                "created": datetime.utcnow().isoformat(),
                "description": description,
                "system_role": system_role,
                "base_prompt": base_prompt,
                "translated": translated,
                "temperature": temperature,
                "top_p": top_p,
                "ii_repeated": ii_repeated,
            },
        )
        next_id = result.scalar()
        self.session.commit()

        # Return as ExperimentsGroup object
        return ExperimentsGroup(
            experiments_group_id=next_id,
            description=description,
            system_role=system_role,
            base_prompt=base_prompt,
            translated=translated,
            temperature=temperature,
            top_p=top_p,
            ii_repeated=ii_repeated,
        )

    def mark_groups_concluded(self, start_id: int, end_id: int) -> int:
        stmt = (
            select(ExperimentsGroup)
            .where(ExperimentsGroup.experiments_group_id >= start_id)
            .where(ExperimentsGroup.experiments_group_id <= end_id)
        )
        groups = list(self.session.exec(stmt))
        for g in groups:
            g.concluded = True
        self.session.commit()
        return len(groups)

    # --- Experiments ---
    def create_experiment(self, exp: ExperimentsList) -> ExperimentsList:
        """Create experiment in the configured schema using raw SQL with auto-increment ID."""
        from datetime import datetime

        from sqlalchemy import text

        # Ensure created timestamp is set
        if exp.created is None:
            exp.created = datetime.utcnow().isoformat()

        # Insert with explicit schema, letting PostgreSQL sequence auto-generate experiment_id
        # Use RETURNING to get the auto-generated experiment_id
        insert_query = text(
            f"""
            INSERT INTO {self.schema}.experiments_list (
                experiments_group_id,
                repeated,
                created,
                questionnaire,
                language_instructions,
                language_questionnaire,
                model_provider,
                model,
                population,
                personality_id,
                repo_sha,
                succeeded
            ) VALUES (
                :experiments_group_id,
                :repeated,
                :created,
                :questionnaire,
                :language_instructions,
                :language_questionnaire,
                :model_provider,
                :model,
                :population,
                :personality_id,
                :repo_sha,
                NULL
            )
            RETURNING experiment_id
        """
        )

        result = self.session.execute(
            insert_query,
            {
                "experiments_group_id": exp.experiments_group_id,
                "repeated": exp.repeated,
                "created": exp.created,
                "questionnaire": exp.questionnaire,
                "language_instructions": exp.language_instructions,
                "language_questionnaire": exp.language_questionnaire,
                "model_provider": exp.model_provider,
                "model": exp.model,
                "population": exp.population,
                "personality_id": exp.personality_id,
                "repo_sha": exp.repo_sha,
            },
        )
        exp.experiment_id = result.scalar()
        self.session.commit()

        return exp

    def next_experiment_id(self) -> int:
        from sqlmodel import func

        result = self.session.exec(select(func.max(ExperimentsList.experiment_id))).first()
        return (result or 0) + 1

    def create_bulk_experiments(
        self, experiments: Iterable[ExperimentsList]
    ) -> List[ExperimentsList]:
        exps = list(experiments)
        for e in exps:
            if e.experiment_id is None:
                e.experiment_id = self.next_experiment_id()
            self.session.add(e)
        self.session.commit()
        for e in exps:
            self.session.refresh(e)
        return exps

    def get_unprocessed(self, experiments_group_id: int) -> List[ExperimentsList]:
        stmt = select(ExperimentsList).where(
            ExperimentsList.experiments_group_id == experiments_group_id
        )
        rows = [e for e in self.session.exec(stmt) if e.succeeded is None]
        rows.sort(key=lambda r: r.experiment_id or 0, reverse=True)
        return rows

    # --- Group processing / status helpers ---
    def mark_group_processed(self, experiments_group_id: int) -> None:
        group = self.session.get(ExperimentsGroup, experiments_group_id)
        if group and not getattr(group, "processed", False):
            group.processed = True  # type: ignore[attr-defined]
            self.session.commit()

    def is_group_complete(self, experiments_group_id: int) -> bool:
        stmt = select(ExperimentsList).where(
            ExperimentsList.experiments_group_id == experiments_group_id
        )
        experiments = list(self.session.exec(stmt))
        if not experiments:
            return False
        return all(e.succeeded is True for e in experiments)

    def check_and_mark_group_processed(self, experiments_group_id: int) -> bool:
        """If all experiments in group are complete, mark group processed.

        Returns True if group marked processed (or already processed)."""
        group = self.session.get(ExperimentsGroup, experiments_group_id)
        if not group:
            return False
        if getattr(group, "processed", False):  # type: ignore[attr-defined]
            return True
        if self.is_group_complete(experiments_group_id):
            group.processed = True  # type: ignore[attr-defined]
            self.session.commit()
            return True
        return False

    def mark_experiment_succeeded(
        self, experiment_id: int, llm_explanation: Optional[str] = None
    ) -> None:
        """Mark experiment as succeeded using schema-aware SQL."""
        from sqlalchemy import text

        update_query = text(
            f"""
            UPDATE {self.schema}.experiments_list
            SET succeeded = TRUE,
                llm_explanation = :llm_explanation
            WHERE experiment_id = :experiment_id
        """
        )

        self.session.execute(
            update_query, {"experiment_id": experiment_id, "llm_explanation": llm_explanation}
        )
        self.session.commit()

    def update_llm_explanation(self, experiment_id: int, llm_explanation: Optional[str]) -> None:
        """Update LLM explanation using schema-aware SQL."""
        from sqlalchemy import text

        update_query = text(
            f"""
            UPDATE {self.schema}.experiments_list
            SET llm_explanation = :llm_explanation
            WHERE experiment_id = :experiment_id
        """
        )

        self.session.execute(
            update_query, {"experiment_id": experiment_id, "llm_explanation": llm_explanation}
        )
        self.session.commit()

    def get_experiment(self, experiment_id: int) -> Optional[ExperimentsList]:
        """Get experiment by ID using schema-aware SQL."""
        from sqlalchemy import text

        query = text(
            f"""
            SELECT * FROM {self.schema}.experiments_list
            WHERE experiment_id = :experiment_id
        """
        )

        result = self.session.execute(query, {"experiment_id": experiment_id})
        row = result.first()

        if row:
            # Convert row to ExperimentsList object
            return ExperimentsList(**dict(row._mapping))
        return None


class FileExperimentsRepository:
    """CSV-backed repository used when ``--no-db`` is set.

    Data is persisted under an ``artifacts/`` directory mirroring the
    ``epqra`` schema tables using CSV format.
    """

    def __init__(self, artifacts_dir: Path | str = "artifacts") -> None:
        self.base = Path(artifacts_dir)
        self.schema_dir = self.base / "personality_trap"
        self.schema_dir.mkdir(parents=True, exist_ok=True)
        self.group_fields = [
            "experiments_group_id",
            "description",
            "system_role",
            "base_prompt",
            "translated",
            "temperature",
            "top_p",
            "ii_repeated",
            "concluded",
            "processed",
            "created",
        ]
        self.exp_fields = [
            "experiment_id",
            "experiments_group_id",
            "repeated",
            "questionnaire",
            "model",
            "succeeded",
            "llm_explanation",
            "created",
        ]

        # Ensure files exist with headers
        self._save_rows([], self.group_fields, self.schema_dir / "experiments_groups")
        self._save_rows([], self.exp_fields, self.schema_dir / "experiments_list")

    # -- helpers ---------------------------------------------------------
    def _load_rows(self, base: Path, fields: List[str]) -> List[dict[str, Any]]:  # noqa: E501
        """Load rows from CSV file."""
        csv_path = base.with_suffix(".csv")
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8") as fp:
                return list(csv.DictReader(fp))  # type: ignore[return-value]
        return []

    def _save_rows(self, rows: List[dict], fields: List[str], base: Path) -> None:
        """Save rows to CSV file for consistent, portable storage."""
        csv_path = base.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    # -- Experiment Groups -----------------------------------------------
    def create_group(
        self,
        description: str,
        system_role: str,
        base_prompt: str,
        translated: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        ii_repeated: bool = False,
    ) -> dict:
        rows = self._load_rows(self.schema_dir / "experiments_groups", self.group_fields)
        new_id = max(int(r["experiments_group_id"]) for r in rows) + 1 if rows else 1
        row = {
            "experiments_group_id": new_id,
            "description": description,
            "system_role": system_role,
            "base_prompt": base_prompt,
            "translated": translated,
            "temperature": temperature,
            "top_p": top_p,
            "ii_repeated": ii_repeated,
            "concluded": False,
            "processed": False,
            "created": datetime.utcnow().isoformat(),
        }
        rows.append(row)
        self._save_rows(rows, self.group_fields, self.schema_dir / "experiments_groups")
        return row

    def mark_groups_concluded(self, start_id: int, end_id: int) -> int:
        rows = self._load_rows(self.schema_dir / "experiments_groups", self.group_fields)
        count = 0
        for r in rows:
            gid = int(r["experiments_group_id"])
            if start_id <= gid <= end_id:
                r["concluded"] = True
                count += 1
        self._save_rows(rows, self.group_fields, self.schema_dir / "experiments_groups")
        return count

    # -- Experiments -----------------------------------------------------
    def create_experiment(self, exp: dict) -> dict:
        rows = self._load_rows(self.schema_dir / "experiments_list", self.exp_fields)
        new_id = max(int(r["experiment_id"]) for r in rows) + 1 if rows else 1
        exp.setdefault("experiment_id", new_id)
        exp.setdefault("succeeded", "")
        exp.setdefault("llm_explanation", "")
        exp.setdefault("created", datetime.utcnow().isoformat())
        rows.append(exp)
        self._save_rows(rows, self.exp_fields, self.schema_dir / "experiments_list")
        return exp

    def next_experiment_id(self) -> int:
        rows = self._load_rows(self.schema_dir / "experiments_list", self.exp_fields)
        return max(int(r["experiment_id"]) for r in rows) + 1 if rows else 1

    def create_bulk_experiments(self, experiments: Iterable[dict]) -> List[dict]:
        exps = list(experiments)
        for e in exps:
            if not e.get("experiment_id"):
                e["experiment_id"] = self.next_experiment_id()
        rows = self._load_rows(self.schema_dir / "experiments_list", self.exp_fields)
        rows.extend(exps)
        self._save_rows(rows, self.exp_fields, self.schema_dir / "experiments_list")
        return exps

    def get_unprocessed(self, experiments_group_id: int) -> List[dict]:
        rows = self._load_rows(self.schema_dir / "experiments_list", self.exp_fields)
        filtered = [
            r
            for r in rows
            if int(r["experiments_group_id"]) == experiments_group_id
            and r.get("succeeded") != "True"
        ]
        filtered.sort(key=lambda r: int(r["experiment_id"]), reverse=True)
        return filtered

    # -- Group processing / status helpers -------------------------------
    def mark_group_processed(self, experiments_group_id: int) -> None:
        rows = self._load_rows(self.schema_dir / "experiments_groups", self.group_fields)
        for r in rows:
            if int(r["experiments_group_id"]) == experiments_group_id:
                r["processed"] = True
        self._save_rows(rows, self.group_fields, self.schema_dir / "experiments_groups")

    def is_group_complete(self, experiments_group_id: int) -> bool:
        rows = self.get_unprocessed(experiments_group_id)
        return not rows

    def check_and_mark_group_processed(self, experiments_group_id: int) -> bool:
        if self.is_group_complete(experiments_group_id):
            self.mark_group_processed(experiments_group_id)
            return True
        return False

    def mark_experiment_succeeded(
        self, experiment_id: int, llm_explanation: Optional[str] = None
    ) -> None:
        rows = self._load_rows(self.schema_dir / "experiments_list", self.exp_fields)
        for r in rows:
            if int(r["experiment_id"]) == experiment_id:
                r["succeeded"] = "True"
                if llm_explanation is not None:
                    r["llm_explanation"] = llm_explanation or ""
        self._save_rows(rows, self.exp_fields, self.schema_dir / "experiments_list")

    def update_llm_explanation(self, experiment_id: int, llm_explanation: Optional[str]) -> None:
        rows = self._load_rows(self.schema_dir / "experiments_list", self.exp_fields)
        for r in rows:
            if int(r["experiment_id"]) == experiment_id:
                r["llm_explanation"] = llm_explanation or ""
        self._save_rows(rows, self.exp_fields, self.schema_dir / "experiments_list")

    def get_experiment(self, experiment_id: int) -> Optional[dict]:
        rows = self._load_rows(self.schema_dir / "experiments_list", self.exp_fields)
        for r in rows:
            if int(r["experiment_id"]) == experiment_id:
                return r
        return None

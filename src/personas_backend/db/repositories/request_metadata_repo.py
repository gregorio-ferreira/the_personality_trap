"""Repositories for experiment request/response payloads."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from personas_backend.db.models import ExperimentRequestMetadata
from personas_backend.db.schema_config import get_experimental_schema
from sqlalchemy import text
from sqlmodel import Session


class ExperimentRequestMetadataRepository:
    """Persist experiment request/response payloads using schema-aware SQL."""

    def __init__(self, session: Session, schema: Optional[str] = None):
        self.session = session
        self.schema = schema or get_experimental_schema()

    def create_entry(
        self,
        *,
        experiment_id: int,
        request_json: Optional[Dict[str, Any]] = None,
        response_json: Optional[Dict[str, Any]] = None,
        request_metadata: Optional[Dict[str, Any]] = None,
    ) -> ExperimentRequestMetadata:
        """Create metadata entry using schema-aware SQL with auto-increment ID."""

        # Convert dicts to JSON strings for JSONB columns
        request_json_str = json.dumps(request_json) if request_json is not None else None
        response_json_str = json.dumps(response_json) if response_json is not None else None
        request_metadata_str = (
            json.dumps(request_metadata) if request_metadata is not None else None
        )

        # Generate created timestamp
        # (Python default_factory doesn't work with raw SQL)
        created_timestamp = datetime.utcnow()

        # Use raw SQL to insert with explicit created timestamp
        insert_query = text(
            f"""
            INSERT INTO {self.schema}.experiment_request_metadata (
                experiment_id,
                created,
                request_json,
                response_json,
                request_metadata
            ) VALUES (
                :experiment_id,
                :created,
                CAST(:request_json AS JSONB),
                CAST(:response_json AS JSONB),
                CAST(:request_metadata AS JSONB)
            )
            RETURNING id, created
        """
        )

        result = self.session.execute(
            insert_query,
            {
                "experiment_id": experiment_id,
                "created": created_timestamp,
                "request_json": request_json_str,
                "response_json": response_json_str,
                "request_metadata": request_metadata_str,
            },
        )
        row = result.first()
        self.session.commit()

        # Return ExperimentRequestMetadata object
        if row:
            return ExperimentRequestMetadata(
                id=row.id,
                created=row.created,
                experiment_id=experiment_id,
                request_json=request_json,
                response_json=response_json,
                request_metadata=request_metadata,
            )
        else:
            raise RuntimeError("Failed to create experiment request metadata")


class FileExperimentRequestMetadataRepository:
    """CSV-backed repository mirroring experiment_request_metadata table."""

    def __init__(self, artifacts_dir: Path | str = "artifacts") -> None:
        self.base = Path(artifacts_dir)
        self.schema_dir = self.base / "personality_trap"
        self.schema_dir.mkdir(parents=True, exist_ok=True)
        self.fields = [
            "id",
            "experiment_id",
            "request_json",
            "response_json",
            "request_metadata",
            "created",
        ]
        self._ensure_file()

    def _path(self) -> Path:
        return self.schema_dir / "experiment_request_metadata.csv"

    def _ensure_file(self) -> None:
        if not self._path().exists():
            with self._path().open("w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=self.fields)
                writer.writeheader()

    def _load_rows(self) -> list[dict[str, Any]]:
        path = self._path()
        if not path.exists():
            return []
        with path.open(newline="", encoding="utf-8") as fp:
            return list(csv.DictReader(fp))

    def _save_rows(self, rows: list[dict[str, Any]]) -> None:
        with self._path().open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=self.fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def create_entry(
        self,
        *,
        experiment_id: int,
        request_json: Optional[Dict[str, Any]] = None,
        response_json: Optional[Dict[str, Any]] = None,
        request_metadata: Optional[Dict[str, Any]] = None,
    ) -> dict[str, Any]:
        rows = self._load_rows()
        new_id = max(int(r["id"]) for r in rows) + 1 if rows else 1
        row = {
            "id": str(new_id),
            "experiment_id": str(experiment_id),
            "request_json": json.dumps(request_json or {}),
            "response_json": json.dumps(response_json or {}),
            "request_metadata": json.dumps(request_metadata or {}),
            "created": datetime.utcnow().isoformat(),
        }
        rows.append(row)
        self._save_rows(rows)
        return row

"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
VENV_SITE = ROOT / ".venv" / "lib" / "python3.12" / "site-packages"

for candidate in (ROOT, SRC, VENV_SITE):
    if candidate.exists() and str(candidate) not in sys.path:  # pragma: no cover
        sys.path.insert(0, str(candidate))

import pytest
from sqlalchemy import types as satypes
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

# Import models so SQLModel metadata is populated for the in-memory engine.
from personas_backend.db import models as db_models  # noqa: E402  pylint: disable=unused-import


@pytest.fixture
def sqlmodel_engine():
    """Provide an in-memory SQLite engine that mirrors the personality_trap schema."""

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    with engine.connect() as connection:
        for table in SQLModel.metadata.tables.values():
            for column in table.c:
                if isinstance(column.type, JSONB):
                    column.type = satypes.JSON()
        connection.exec_driver_sql("ATTACH DATABASE ':memory:' AS personality_trap")
        SQLModel.metadata.create_all(connection)

    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def session_factory(sqlmodel_engine) -> Callable[[], Iterator[Session]]:
    """Context manager factory yielding SQLModel sessions bound to the test engine."""

    @contextmanager
    def factory() -> Iterator[Session]:
        with Session(sqlmodel_engine) as session:  # type: ignore[call-arg]
            yield session

    return factory


@pytest.fixture
def patch_get_session(monkeypatch: pytest.MonkeyPatch, session_factory):
    """Redirect ``personas_backend.db.session.get_session`` to the in-memory engine."""

    from personas_backend.db import experiment_groups, experiments, session as session_module

    monkeypatch.setattr(session_module, "get_session", session_factory)
    monkeypatch.setattr(experiments, "get_session", session_factory)
    monkeypatch.setattr(experiment_groups, "get_session", session_factory)
    return session_factory

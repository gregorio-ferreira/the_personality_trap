"""Helpers for creating SQLModel sessions with environment-based configuration."""

from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Callable, Iterator, TypeVar

from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine


def _build_database_url() -> str:
    # Prefer explicit DATABASE_URL env var; fallback to discrete env pieces
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    host = os.getenv("PERSONAS_PG__HOST", "localhost")
    port = os.getenv("PERSONAS_PG__PORT", "5432")
    db = os.getenv("PERSONAS_PG__DATABASE", "personas")
    user = os.getenv("PERSONAS_PG__USER", "personas")
    password = os.getenv("PERSONAS_PG__PASSWORD", "personas")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


@lru_cache(maxsize=1)
def get_engine(echo: bool = False) -> Engine:
    return create_engine(_build_database_url(), echo=echo, future=True)


def init_db(echo: bool = False) -> None:
    engine = get_engine(echo=echo)
    # Import models to register metadata
    from personas_backend.db import models  # noqa: F401

    SQLModel.metadata.create_all(engine)


@contextmanager
def get_session() -> Iterator[Session]:  # contextmanager wrapper
    engine = get_engine()
    with Session(engine) as session:  # type: ignore[call-arg]
        yield session


F = TypeVar("F", bound=Callable[..., Any])


def with_session(fn: F) -> F:  # type: ignore[misc]
    """Decorator to inject a managed session (kwarg 'session')."""

    def wrapper(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        with Session(get_engine()) as session:  # type: ignore[call-arg]
            return fn(*args, session=session, **kwargs)

    return wrapper  # type: ignore[return-value]

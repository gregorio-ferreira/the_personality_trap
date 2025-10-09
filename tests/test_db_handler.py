import os

from personas_backend.db.db_handler import (  # type: ignore
    DatabaseHandler,
)
from personas_backend.utils.config import (  # type: ignore
    ConfigManager,
)


class DummyEngine:  # pragma: no cover - simple stub
    def __init__(self, url: str):  # type: ignore[no-untyped-def]
        self.url = url

    def connect(self):  # type: ignore[no-untyped-def]
        return self

    def dispose(self):  # type: ignore[no-untyped-def]
        pass


class _ConnectionContext:  # pragma: no cover - helper for context manager protocol
    def __init__(self, connection):  # type: ignore[no-untyped-def]
        self._connection = connection

    def __enter__(self):  # type: ignore[no-untyped-def]
        return self._connection

    def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
        return False


class RecordingResult:  # pragma: no cover - simple scalar holder
    def __init__(self, value):  # type: ignore[no-untyped-def]
        self._value = value

    def scalar_one_or_none(self):  # type: ignore[no-untyped-def]
        return self._value


class RecordingConnection:  # pragma: no cover - captures executed statements
    def __init__(self, scalar_results=None):  # type: ignore[no-untyped-def]
        self.scalar_results = list(scalar_results or [])
        self.executed = []

    def execute(self, statement, parameters=None):  # type: ignore[no-untyped-def]
        self.executed.append((statement, parameters))
        # Simulate SELECT results for schema existence checks
        value = self.scalar_results.pop(0) if self.scalar_results else None
        return RecordingResult(value)


class FakeEngine:  # pragma: no cover - injectable engine stub
    def __init__(self, connection, url="postgresql://stub:stub@localhost:5432/stub"):  # type: ignore[no-untyped-def]
        self._connection = connection
        self.url = url

    def connect(self):  # type: ignore[no-untyped-def]
        return _ConnectionContext(self._connection)

    def dispose(self):  # type: ignore[no-untyped-def]
        pass


def test_db_handler_connection_url(monkeypatch, tmp_path):  # type: ignore
    # Prepare config file with non-default port
    cfg_file = tmp_path / "db.yaml"
    cfg_file.write_text(
        (
            "pg:\n"
            "  host: test-host\n"
            "  database: testdb\n"
            "  user: testuser\n"
            "  password: testpass\n"
            "  port: 6544\n"
        ),
        encoding="utf-8",
    )

    urls: list[str] = []

    def fake_create_engine(url: str, *_, **__):  # type: ignore[no-untyped-def]
        urls.append(url)
        return DummyEngine(url)

    import personas_backend.db.db_handler as dbh  # type: ignore

    monkeypatch.setattr(dbh, "create_engine", fake_create_engine)

    cm = ConfigManager(config_path=str(cfg_file))
    # Instantiation should invoke create_engine
    DatabaseHandler(config_manager=cm)
    assert urls, "create_engine was not called"
    assert urls[0] == "postgresql://testuser:testpass@test-host:6544/testdb"


def _make_config(tmp_path):  # type: ignore[no-untyped-def]
    cfg_file = tmp_path / "db.yaml"
    cfg_file.write_text(
        (
            "pg:\n"
            "  host: test-host\n"
            "  database: testdb\n"
            "  user: testuser\n"
            "  password: testpass\n"
            "  port: 5432\n"
        ),
        encoding="utf-8",
    )
    return ConfigManager(config_path=str(cfg_file))


def test_schema_exists(monkeypatch, tmp_path):  # type: ignore
    responses = RecordingConnection(scalar_results=[1])
    engine = FakeEngine(responses)

    import personas_backend.db.db_handler as dbh  # type: ignore

    monkeypatch.setattr(dbh.DatabaseHandler, "connect_to_postgres", lambda self, port=None: engine)

    handler = DatabaseHandler(config_manager=_make_config(tmp_path))

    assert handler.schema_exists("foo_schema") is True
    assert responses.executed
    statement, params = responses.executed[0]
    assert params == {"schema_name": "foo_schema"}
    assert "schema" in str(statement).lower()


def test_run_migrations_sets_env_and_invokes_alembic(monkeypatch, tmp_path):  # type: ignore
    engine = FakeEngine(RecordingConnection())

    import personas_backend.db.db_handler as dbh  # type: ignore

    monkeypatch.setattr(dbh.DatabaseHandler, "connect_to_postgres", lambda self, port=None: engine)

    handler = DatabaseHandler(config_manager=_make_config(tmp_path))

    called = {}

    def fake_upgrade(config, revision):  # type: ignore[no-untyped-def]
        called["revision"] = revision
        called["url"] = config.get_main_option("sqlalchemy.url")
        called["script_location"] = config.get_main_option("script_location")

    monkeypatch.setattr(dbh.command, "upgrade", fake_upgrade)

    previous = os.environ.get("PERSONAS_TARGET_SCHEMA")
    os.environ["PERSONAS_TARGET_SCHEMA"] = "existing"
    try:
        handler.run_migrations(schema="run_schema", revision="head")
        assert called
        assert called["revision"] == "head"
        assert called["url"] == engine.url
        assert called["script_location"].endswith("alembic")
        assert os.environ["PERSONAS_TARGET_SCHEMA"] == "existing"
    finally:
        if previous is None:
            os.environ.pop("PERSONAS_TARGET_SCHEMA", None)
        else:
            os.environ["PERSONAS_TARGET_SCHEMA"] = previous

import textwrap
from pathlib import Path

from personas_backend.utils.config import ConfigManager


def write_temp_config(tmp_path: Path, content: str) -> Path:
    file_path = tmp_path / "temp.yaml"
    file_path.write_text(textwrap.dedent(content), encoding="utf-8")
    return file_path


def test_env_override_openai(tmp_path, monkeypatch):  # type: ignore
    cfg_path = write_temp_config(
        tmp_path,
        """
        openai:
          api_key: BASE_KEY
        pg:
          host: localhost
          database: personas
          user: personas
          password: secret
          port: 5432
        """,
    )
    monkeypatch.setenv("PERSONAS_OPENAI__API_KEY", "OVERRIDE_KEY")
    cm = ConfigManager(config_path=str(cfg_path))
    assert cm.openai_config.api_key == "OVERRIDE_KEY"


def test_postgres_port_normalization(tmp_path):  # type: ignore
    cfg_path = write_temp_config(
        tmp_path,
        """
        pg:
          host: localhost
          database: personas
          user: personas
          password: secret
          port: "6543"
        """,
    )
    cm = ConfigManager(config_path=str(cfg_path))
    assert cm.pg_config.port == 6543


def test_get_helper(tmp_path):  # type: ignore
    cfg_path = write_temp_config(
        tmp_path,
        """
        openai:
          api_key: SOMEKEY
        """,
    )
    cm = ConfigManager(config_path=str(cfg_path))
    assert cm.get("openai", "api-key") == "SOMEKEY"
    assert cm.get("missing", "key", default="fallback") == "fallback"

#!/usr/bin/env bash
# Codex setup script for the_personality_trap repository.
# Pattern inspired by JSON-driven runner (bryanschmidty/openai-codex-setupscript) but simplified.
# Executes sections defined in embedded JSON. Shows only headers unless DRY_RUN=true.

set -euo pipefail

read -r -d '' CONFIG_JSON <<'JSONCFG'
{
  "variables": [
    "PERSONAS_OPENAI__API_KEY",
    "PERSONAS_PG__HOST",
    "PERSONAS_PG__DATABASE",
    "PERSONAS_PG__USER",
    "PERSONAS_PG__PASSWORD"
  ],
  "sections": [
    {"desc": "Validate required variables", "cmds": [
      "echo 'All required env vars present.'"
    ]},
    {"desc": "System packages", "cmds": [
      "sudo apt-get update -y",
      "sudo apt-get install -y build-essential jq make curl libpq-dev pkg-config"
    ]},
    {"desc": "Install uv", "cmds": [
      "curl -LsSf https://astral.sh/uv/install.sh | sh",
      "export PATH=$HOME/.local/bin:$PATH",
      "uv --version"
    ]},
    {"desc": "Python deps sync", "cmds": [
      "export PATH=$HOME/.local/bin:$PATH",
      "if [ -f uv.lock ]; then uv sync --all-extras --dev; else pip install --upgrade pip && pip install .[dev]; fi"
    ]},
    {"desc": "Migrations (non-fatal)", "cmds": [
      "if [ -n \"${PERSONAS_PG__HOST:-}\" ]; then make db-upgrade || echo 'Migration step failed (non-fatal)'; else echo 'Skipping migrations (no DB host)'; fi"
    ]},
    {"desc": "Smoke test imports", "cmds": [
      "python - <<'PY'\nimport personas_backend\nfrom personas_backend.utils.config import ConfigManager\nprint('Imports OK')\nPY"
    ]}
  ]
}
JSONCFG

if ! command -v jq >/dev/null 2>&1; then
  echo "jq missing; attempting install..." >&2
  sudo apt-get update -y && sudo apt-get install -y jq
fi

missing=0
echo "$CONFIG_JSON" | jq -r '.variables[]' | while read -r v; do
  if [ -z "${!v:-}" ]; then
    echo "[MISSING] $v" >&2
    missing=1
  fi
done
# shellcheck disable=SC2154
if [ "$missing" -ne 0 ]; then
  echo "One or more required environment variables are missing." >&2
  exit 1
fi

dr="${DRY_RUN:-false}";

echo "$CONFIG_JSON" | jq -c '.sections[]' | while read -r section; do
  desc=$(echo "$section" | jq -r '.desc')
  echo "==> $desc"
  echo "$section" | jq -r '.cmds[]' | while read -r cmd; do
    if [ "$dr" = "true" ]; then
      echo "DRY_RUN: $cmd"
    else
      set +e
      eval "$cmd" >/tmp/codex_section_out 2>&1
      status=$?
      set -e
      if [ $status -ne 0 ]; then
        echo "[FAIL] $desc" >&2
        cat /tmp/codex_section_out >&2
        exit $status
      fi
    fi
  done
  echo "[OK] $desc"
done

echo "Codex setup completed successfully."

"""Generic Alembic revision script."""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = ${repr(revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:  # noqa: D401
    ${upgrades if upgrades else 'pass'}


def downgrade() -> None:  # noqa: D401
    ${downgrades if downgrades else 'pass'}

"""v10_initial_schema — Baseline for existing V9 database.

This migration is intentionally a no-op for existing SQLite databases.
It stamps the current schema as the starting point for future migrations.

For fresh PostgreSQL installs, use db.models.create_all_tables(engine) instead.

Revision ID: 651e01fa3df3
Revises:
Create Date: 2026-03-19
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '651e01fa3df3'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Stamp existing database as baseline. No schema changes."""
    # Existing V9 databases already have all tables.
    # This migration exists to establish the Alembic version tracking baseline.
    #
    # For fresh installs: db.models.create_all_tables(engine) creates everything.
    # For upgrades: future migrations will add V10-specific schema changes.
    pass


def downgrade() -> None:
    """Cannot downgrade past baseline."""
    pass

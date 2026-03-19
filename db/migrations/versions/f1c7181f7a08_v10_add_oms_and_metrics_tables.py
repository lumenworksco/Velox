"""v10_add_oms_and_metrics_tables

Revision ID: f1c7181f7a08
Revises: 651e01fa3df3
Create Date: 2026-03-19 19:19:04.888776

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f1c7181f7a08'
down_revision: Union[str, Sequence[str], None] = '651e01fa3df3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add V10 OMS order log, circuit breaker events, and event log tables."""

    # OMS persistent order log
    op.create_table(
        'oms_orders',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('oms_id', sa.String(12), nullable=False, unique=True),
        sa.Column('broker_order_id', sa.Text, server_default=''),
        sa.Column('idempotency_key', sa.Text, nullable=False, server_default='', unique=True),
        sa.Column('symbol', sa.String(10), nullable=False),
        sa.Column('strategy', sa.String(30), nullable=False),
        sa.Column('side', sa.String(4), nullable=False),
        sa.Column('order_type', sa.String(10)),
        sa.Column('qty', sa.Integer),
        sa.Column('limit_price', sa.Float),
        sa.Column('take_profit', sa.Float),
        sa.Column('stop_loss', sa.Float),
        sa.Column('state', sa.String(15), nullable=False),
        sa.Column('filled_qty', sa.Integer, server_default='0'),
        sa.Column('filled_avg_price', sa.Float, server_default='0.0'),
        sa.Column('created_at', sa.Text),
        sa.Column('submitted_at', sa.Text),
        sa.Column('filled_at', sa.Text),
        sa.Column('cancelled_at', sa.Text),
        sa.Column('pair_id', sa.Text, server_default=''),
    )
    op.create_index('idx_oms_symbol', 'oms_orders', ['symbol'])
    op.create_index('idx_oms_state', 'oms_orders', ['state'])
    op.create_index('idx_oms_created', 'oms_orders', ['created_at'])

    # Circuit breaker event log
    op.create_table(
        'circuit_breaker_events',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('timestamp', sa.Text, nullable=False),
        sa.Column('old_tier', sa.String(10)),
        sa.Column('new_tier', sa.String(10), nullable=False),
        sa.Column('day_pnl_pct', sa.Float),
        sa.Column('equity', sa.Float),
    )

    # Event bus persistent log (critical events only)
    op.create_table(
        'event_log',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('timestamp', sa.Text, nullable=False),
        sa.Column('event_type', sa.String(50), nullable=False),
        sa.Column('source', sa.String(30)),
        sa.Column('data_json', sa.Text),
    )
    op.create_index('idx_event_type', 'event_log', ['event_type'])
    op.create_index('idx_event_timestamp', 'event_log', ['timestamp'])

    # Add cost tracking columns to trades table
    with op.batch_alter_table('trades') as batch_op:
        batch_op.add_column(sa.Column('estimated_cost_bps', sa.Float, nullable=True))
        batch_op.add_column(sa.Column('actual_slippage_bps', sa.Float, nullable=True))


def downgrade() -> None:
    """Remove V10 tables and columns."""
    op.drop_table('event_log')
    op.drop_table('circuit_breaker_events')
    op.drop_table('oms_orders')

    with op.batch_alter_table('trades') as batch_op:
        batch_op.drop_column('actual_slippage_bps')
        batch_op.drop_column('estimated_cost_bps')

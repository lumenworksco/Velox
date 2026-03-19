"""V10 Database — SQLAlchemy abstraction supporting both SQLite and PostgreSQL.

Usage:
    from db import get_engine, get_session

    # For queries:
    with get_session() as session:
        result = session.execute(text("SELECT * FROM trades"))

    # Switch backends via DATABASE_URL env var:
    #   SQLite (default):   sqlite:///bot.db
    #   PostgreSQL:         postgresql://user:pass@localhost:5432/velox
"""

import os
import logging
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, StaticPool

import config

logger = logging.getLogger(__name__)

# Default to SQLite for backward compatibility
_DEFAULT_URL = f"sqlite:///{config.DB_FILE}"
DATABASE_URL = os.getenv("DATABASE_URL", _DEFAULT_URL)

_engine = None
_SessionFactory = None


def get_engine():
    """Get or create the SQLAlchemy engine (singleton)."""
    global _engine
    if _engine is None:
        is_sqlite = DATABASE_URL.startswith("sqlite")

        if is_sqlite:
            _engine = create_engine(
                DATABASE_URL,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,  # Single connection for SQLite
                echo=False,
            )
            # Enable WAL mode for SQLite
            with _engine.connect() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL"))
                conn.execute(text("PRAGMA synchronous=NORMAL"))
                conn.commit()
        else:
            _engine = create_engine(
                DATABASE_URL,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,  # Verify connections before use
                echo=False,
            )

        logger.info(f"Database engine created: {'SQLite' if is_sqlite else 'PostgreSQL'}")
    return _engine


def get_session_factory():
    """Get or create the session factory."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())
    return _SessionFactory


@contextmanager
def get_session():
    """Context manager for database sessions with automatic commit/rollback."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

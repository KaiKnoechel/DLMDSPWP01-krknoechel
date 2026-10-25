"""Unit tests for find_ideal_func.database."""

from __future__ import annotations

from sqlalchemy import inspect

from find_ideal_func.database import DbManager, Mappings


def test_init_db_creates_all_tables(tmp_path):
    """Test that initializing the database creates all required tables."""
    db = DbManager(f"sqlite:///{tmp_path / 'db.sqlite'}")
    db.init_db()

    tables = set(inspect(db.engine).get_table_names())
    assert {"TrainingData", "IdealFunction", "Mappings"} <= tables


def test_get_session_can_insert_and_query(tmp_path):
    """Test that a session can insert and query the Mappings table."""
    db = DbManager(f"sqlite:///{tmp_path / 'db.sqlite'}")
    db.init_db()
    session = db.get_session()

    session.add(Mappings(x=1.0, y=2.0, delta_y=0.1, ideal_function_number=3))
    session.commit()

    result = session.query(Mappings).one()
    assert result.ideal_function_number == 3
    session.close()

"""SQLAlchemy  schema for the database and engine management.

This module defines the SQLAlchemy schema for the database used in the find_ideal_func module. It includes the definition of the IdealFunctionColumns model, which represents an ideal function in the database, and provides functions to manage the database engine and session.

Required tables:
- IdealFunction: x and 50 ideal function columns
- TrainingData: x, y1, y2, y3, y4 columns
- Mappings: x, y, delta_y, ideal_function_number of matched test points 
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Column, Float, Integer, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""


class TData(Base):
    """SQLAlchemy model for the TrainingData table."""

    __tablename__ = "TrainingData"

    x = Column(Float, primary_key=True, nullable=False)
    y1 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)
    y3 = Column(Float, nullable=False)
    y4 = Column(Float, nullable=False)


class Mappings(Base):
    """SQLAlchemy model for the Mappings table."""

    __tablename__ = "Mappings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    delta_y = Column(Float, nullable=False)
    ideal_function_number = Column(Integer, nullable=False)


class DbManager:
    """
    Manages the database engine and session for the find_ideal_func module.

    This class provides methods to create a database engine, manage sessions, and initialize the database schema.
    """

    def __init__(self, db_url: str):
        """
        Initialize the DbManager with a database URL.

        Args:
            db_url (str): The database URL for SQLAlchemy.
        """
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def init_db(self):
        """Create all tables in the database."""
        Base.metadata.create_all(bind=self.engine)

    def get_session(self) -> Session:
        """
        Get a new SQLAlchemy session.

        Returns:
            Session: A new SQLAlchemy session.
        """
        return self.SessionLocal()

    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        """Provide a transactional scope around a series of operations."""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()


def _ideal_function_columns() -> dict[str, Column]:
    """Return a dictionary of ideal function columns."""
    return {f"y{i}": Column(Float, nullable=False) for i in range(1, 51)}


IdealFunctionColumns = type(
    "IdealFunctionColumns",
    (Base,),
    {
        "__tablename__": "IdealFunction",
        "x": Column(Float, primary_key=True, nullable=False),
        **_ideal_function_columns(),
    },
)

"""
Database Connection & Session Management
=========================================
Configures SQLite database connection using SQLAlchemy.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Default database location inside data/ directory
DB_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "app_workspace.db")

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False},  # Needed for Streamlit multi-threading
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db():
    """Create all database tables if they do not exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Context manager generator for DB sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
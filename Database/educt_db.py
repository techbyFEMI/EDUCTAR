from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
from contextlib import contextmanager

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    # Fallback to SQLite if DATABASE_URL is not set in environment
    DATABASE_URL = "sqlite:///./eductar.db"
elif DATABASE_URL.startswith("postgres://"):
    # SQLAlchemy requires postgresql:// instead of legacy postgres://
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine_kwargs = {
    "echo": False,
    "pool_pre_ping": True,
    "connect_args": connect_args,
}

if not DATABASE_URL.startswith("sqlite"):
    engine_kwargs.update({
        "pool_recycle": 300,
        "pool_size": 10,
        "max_overflow": 20,
    })

engine = create_engine(DATABASE_URL, **engine_kwargs)

sessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


Base = declarative_base()

def init_db():
    try:
        Base.metadata.create_all(bind=engine)
        print(">> Database tables initialized successfully.")
    except Exception as e:
        print(f">> Warning: Database initialization failed: {e}")

@contextmanager
def get_db():
    db = sessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
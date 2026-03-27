from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL, echo=True, pool_pre_ping=True,pool_recycle=300)

sessionLocal = sessionmaker(bind=engine, autocommit=False,autoflush=False,)

Base = declarative_base()
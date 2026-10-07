import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.models.database import Base

DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+psycopg2://seismo_user:seismo_password@localhost:5432/seismodetect"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)
    print("PostgreSQL tables created successfully.")

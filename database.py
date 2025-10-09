from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "root")
DB_NAME = os.getenv("DB_NAME", "automl")
DB_PORT = os.getenv("DB_PORT", "5432")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Database Models
class AutoMLRun(Base):
    __tablename__ = "automl_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    target_column = Column(String, nullable=False)
    top_model = Column(String, nullable=False)
    model_path = Column(String, nullable=False)
    processing_time = Column(Float, nullable=False)
    recommendations = Column(JSON, nullable=False)  # Store the full recommendations array
    created_at = Column(DateTime, default=datetime.utcnow)
    
class ModelMetrics(Base):
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, nullable=False)  # Foreign key to automl_runs
    model_name = Column(String, nullable=False)
    accuracy = Column(Float, nullable=False)
    auc = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    train_time = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
def create_tables():
    Base.metadata.create_all(bind=engine)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
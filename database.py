from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON , Boolean ,  ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker , relationship
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


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    profession = Column(String, nullable=True)          
    company = Column(String, nullable=True)             
    profile_image = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship with AutoML runs
    automl_runs = relationship("AutoMLRun", back_populates="user", cascade="all, delete-orphan")

    cleaning_sessions = relationship("DataCleaningSession", back_populates="user", cascade="all, delete-orphan")


# Database Models
class AutoMLRun(Base):
    __tablename__ = "automl_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    filename = Column(String, nullable=False)
    target_column = Column(String, nullable=False)
    top_model = Column(String, nullable=False)
    model_path = Column(String, nullable=False)
    processing_time = Column(Float, nullable=False)
    recommendations = Column(JSON, nullable=False)  # Store the full recommendations array
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="automl_runs")

    metrics = relationship("ModelMetrics", back_populates="run", cascade="all, delete-orphan")
    
class ModelMetrics(Base):
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    # run_id = Column(Integer, nullable=False)  # Foreign key to automl_runs
    run_id = Column(Integer, ForeignKey('automl_runs.id', ondelete='CASCADE'), nullable=False)
    model_name = Column(String, nullable=False)
    accuracy = Column(Float, nullable=False)
    auc = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    train_time = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("AutoMLRun", back_populates="metrics")


class DataCleaningSession(Base):
    """Model to track data cleaning sessions"""
    __tablename__ = "data_cleaning_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    original_shape = Column(String(50))  # e.g., "(1000, 20)"
    cleaned_shape = Column(String(50))
    cleaning_method = Column(String(20))  # "manual" or "automatic"
    operations_applied = Column(Text)  # JSON string or text description
    status = Column(String(20), default="uploaded")  # uploaded, cleaning, cleaned, failed
    upload_date = Column(DateTime, default=datetime.utcnow)
    cleaned_date = Column(DateTime, nullable=True)
    
    # Relationship
    user = relationship("User", back_populates="cleaning_sessions")

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
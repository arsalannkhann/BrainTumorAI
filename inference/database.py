"""
Database models and session management for Brain Tumor AI Platform.
Uses SQLAlchemy with SQLite.
"""

import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLITE_URL = "sqlite:///./brain_tumor_history.db"

engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class PredictionRecord(Base):
    """
    Record of a single inference run.
    Stores metadata, classification results, and paths to saved files.
    """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # File Paths (relative to backend root)
    image_path = Column(String, nullable=True)     # Original uploaded image
    mask_path = Column(String, nullable=True)      # Generated segmentation mask
    gradcam_path = Column(String, nullable=True)   # Generated info/heatmap
    
    # Classification Results
    predicted_class = Column(String, nullable=True)
    confidence_score = Column(Float, nullable=True)
    probabilities = Column(JSON, nullable=True)  # Stored as JSON dict
    
    # Segmentation Stats
    tumor_area_percentage = Column(Float, nullable=True)
    has_tumor = Column(Boolean, default=False)
    
    # Status
    validation_passed = Column(Boolean, default=True)


def init_db():
    """Initialize the database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

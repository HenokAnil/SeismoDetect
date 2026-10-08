import logging
from sqlalchemy import text
from backend.api.db_session import SessionLocal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate():
    db = SessionLocal()
    try:
        # events table
        db.execute(text("ALTER TABLE events ADD COLUMN IF NOT EXISTS location_source VARCHAR;"))
        db.execute(text("ALTER TABLE events ADD COLUMN IF NOT EXISTS location_confidence FLOAT;"))
        
        # generated_reports table
        db.execute(text("ALTER TABLE generated_reports ADD COLUMN IF NOT EXISTS context_json JSONB;"))
        db.execute(text("ALTER TABLE generated_reports ADD COLUMN IF NOT EXISTS pipeline_version VARCHAR;"))
        db.execute(text("ALTER TABLE generated_reports ADD COLUMN IF NOT EXISTS embedding_model VARCHAR;"))
        
        db.commit()
        logger.info("Successfully added missing columns to the database.")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    migrate()

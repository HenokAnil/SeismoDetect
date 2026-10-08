from sqlalchemy import Column, String, Float, DateTime, Boolean, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
import uuid

Base = declarative_base()

class Signal(Base):
    __tablename__ = 'signals'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    network = Column(String, nullable=False)
    station = Column(String, nullable=False)
    channel = Column(String, nullable=False)
    location = Column(String)
    starttime = Column(DateTime(timezone=True), nullable=False)
    endtime = Column(DateTime(timezone=True), nullable=False)
    sample_rate = Column(Float, nullable=False)
    s3_key = Column(String, nullable=False, unique=True)
    source = Column(String)
    ingested_at = Column(DateTime(timezone=True), server_default=func.now())
    quality = relationship("SignalQuality", back_populates="signal", uselist=False)

class Station(Base):
    __tablename__ = 'stations'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    network = Column(String)
    station = Column(String)
    lat = Column(Float)
    lon = Column(Float)
    elevation = Column(Float)
    sensor_type = Column(String)

class SignalQuality(Base):
    __tablename__ = 'signal_quality'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    signal_id = Column(UUID(as_uuid=True), ForeignKey('signals.id'))
    snr = Column(Float)
    rms = Column(Float)
    noise_label = Column(String)
    clipping_flag = Column(Boolean, default=False)
    quality_score = Column(Float)
    features_json = Column(JSONB)
    computed_at = Column(DateTime(timezone=True), server_default=func.now())
    signal = relationship("Signal", back_populates="quality")

class Event(Base):
    __tablename__ = 'events'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    signal_id = Column(UUID(as_uuid=True), ForeignKey('signals.id'))
    origin_time = Column(DateTime(timezone=True))
    detection_prob = Column(Float)
    magnitude_proxy = Column(Float)
    model_version = Column(String)
    lat = Column(Float)
    lon = Column(Float)
    depth = Column(Float)
    location_source = Column(String)
    location_confidence = Column(Float)
    catalog_source = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Pick(Base):
    __tablename__ = 'picks'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_id = Column(UUID(as_uuid=True), ForeignKey('events.id'))
    signal_id = Column(UUID(as_uuid=True), ForeignKey('signals.id'))
    phase = Column(String)
    pick_time = Column(DateTime(timezone=True))
    confidence = Column(Float)
    uncertainty_s = Column(Float)
    model_version = Column(String)

class ProcessingRun(Base):
    __tablename__ = 'processing_runs'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    signal_id = Column(UUID(as_uuid=True), ForeignKey('signals.id'))
    run_type = Column(String)
    status = Column(String, default='pending')
    started_at = Column(DateTime(timezone=True))
    finished_at = Column(DateTime(timezone=True))
    config_json = Column(JSONB)
    error_msg = Column(String)


class EventMetadata(Base):
    """
    Cache table for authoritative external event metadata (USGS, ShakeMap, PAGER).
    Keyed by usgs_id so repeated lookups never re-hit the USGS API.
    """
    __tablename__ = 'events_metadata'
    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # FK to our internal event — nullable because some USGS events may be
    # historical references without a matching local detection.
    event_id     = Column(UUID(as_uuid=True), ForeignKey('events.id'), nullable=True)
    usgs_id      = Column(String, unique=True, index=True, nullable=False)
    magnitude    = Column(Float)
    depth_km     = Column(Float)
    lat          = Column(Float)
    lon          = Column(Float)
    place        = Column(String)
    origin_time  = Column(DateTime(timezone=True))
    felt         = Column(Integer)
    shakemap_url = Column(String)
    pager_alert  = Column(String)   # 'green' | 'yellow' | 'orange' | 'red' | None
    raw_json     = Column(JSONB)    # full USGS GeoJSON blob for reproducibility
    fetched_at   = Column(DateTime(timezone=True), server_default=func.now())


class GeneratedReport(Base):
    """
    Stores every RAG-generated report: status tracking, narrative text,
    numeric forecast, list of retrieved similar event IDs, and MinIO artefact paths.
    """
    __tablename__ = 'generated_reports'
    id             = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_id       = Column(UUID(as_uuid=True), ForeignKey('events.id'), nullable=False)
    status         = Column(String, default='pending')   # pending | running | done | error
    model_version  = Column(String)
    narrative_text = Column(Text)
    forecast_json  = Column(JSONB)
    context_json   = Column(JSONB)
    pipeline_version = Column(String)
    embedding_model = Column(String)
    similar_ids    = Column(JSONB)   # list of retrieved event UUIDs
    # MinIO artefact paths
    pdf_s3_key     = Column(String)
    html_s3_key    = Column(String)
    json_s3_key    = Column(String)
    error_msg      = Column(Text)
    generated_at   = Column(DateTime(timezone=True), server_default=func.now())
    updated_at     = Column(DateTime(timezone=True), onupdate=func.now())

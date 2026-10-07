"""
backend/reporting — RAG-powered seismic event report generation package.

Modules:
  usgs_client       — fetch & cache USGS event GeoJSON in Postgres
  numeric_forecast  — GMPE-based PGA/MMI predictions + optional XGBoost regressor
  context_builder   — assemble enriched RAG context dict from hits + metadata
  rag_generator     — T5-based narrative generation with anti-hallucination checks
  storage           — upload PDF/HTML/JSON artefacts to MinIO
  orchestrator      — end-to-end pipeline called by the FastAPI background task
"""

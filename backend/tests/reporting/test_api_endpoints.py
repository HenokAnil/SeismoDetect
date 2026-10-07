"""
tests/reporting/test_api_endpoints.py
----------------------------------------
Integration-style tests for the reporting API endpoints.
Qdrant, MinIO, Postgres, and the orchestrator pipeline are all mocked.
Tests use FastAPI TestClient and run without any external services.
"""

import uuid
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_event():
    """Return a minimal ORM-like event object."""
    ev = MagicMock()
    ev.id = uuid.uuid4()
    ev.signal_id = uuid.uuid4()
    ev.detection_prob = 0.91
    ev.magnitude_proxy = 5.3
    ev.origin_time = None
    ev.depth = 12.0
    ev.lat = None
    ev.lon = None
    ev.model_version = "EQTransformer v1.0"
    ev.catalog_source = "ml_pipeline"
    return ev


def _make_fake_report(status="pending"):
    row = MagicMock()
    row.id = uuid.uuid4()
    row.event_id = uuid.uuid4()
    row.status = status
    row.model_version = "t5-small"
    row.narrative_text = "Test narrative." if status == "done" else None
    row.forecast_json = {"pga_g_median": 0.12} if status == "done" else None
    row.similar_ids = [] if status == "done" else None
    row.json_s3_key = "reports/test/report.json" if status == "done" else None
    row.html_s3_key = "reports/test/report.html" if status == "done" else None
    row.pdf_s3_key = None
    row.error_msg = None
    row.generated_at = None
    return row


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _make_mock_db(event, report=None):
    """Return a mocked DB session."""
    mock_db = MagicMock()
    
    mock_filter = MagicMock()
    mock_filter.first.return_value = event
    mock_filter.order_by.return_value.first.return_value = report
    mock_db.query.return_value.filter.return_value = mock_filter
    return mock_db

@pytest.fixture
def client():
    """Create a TestClient with all heavy singletons mocked at import time."""
    with (
        patch("backend.api.minio_client.MinIOStorage.__init__", return_value=None),
        patch("backend.api.minio_client.MinIOStorage.ensure_bucket"),
        patch("backend.api.main.SimilarityIndex"),
        patch("backend.api.main.EmbeddingPipeline"),
        patch("backend.api.main.EQTransformerDetector"),
        patch("backend.api.main.NoiseClassifier"),
        patch("backend.api.main.init_db"),
    ):
        from backend.api.main import app, get_db
        # We must yield the app so we can set dependency overrides in tests
        yield app, get_db


def test_post_report_returns_202_when_event_exists(client):
    app, get_db = client
    fake_event = _make_fake_event()
    mock_db = _make_mock_db(fake_event)
    app.dependency_overrides[get_db] = lambda: mock_db

    with patch("backend.api.main.run_report_pipeline") as mock_pipeline:
        with TestClient(app) as test_client:
            response = test_client.post(f"/api/events/{fake_event.id}/report")

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "accepted"
    assert "report_id" in data
    assert "poll_url" in data
    
    app.dependency_overrides.clear()


def test_post_report_returns_404_for_missing_event(client):
    app, get_db = client
    mock_db = _make_mock_db(None)
    app.dependency_overrides[get_db] = lambda: mock_db

    with TestClient(app) as test_client:
        response = test_client.post(f"/api/events/{uuid.uuid4()}/report")

    assert response.status_code == 404
    app.dependency_overrides.clear()


def test_get_report_returns_pending_status(client):
    app, get_db = client
    fake_event  = _make_fake_event()
    fake_report = _make_fake_report(status="pending")
    mock_db = _make_mock_db(fake_event, fake_report)
    app.dependency_overrides[get_db] = lambda: mock_db

    with TestClient(app) as test_client:
        response = test_client.get(f"/api/events/{fake_event.id}/report")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "pending"
    app.dependency_overrides.clear()


def test_get_report_returns_full_content_when_done(client):
    app, get_db = client
    fake_event  = _make_fake_event()
    fake_report = _make_fake_report(status="done")
    mock_db = _make_mock_db(fake_event, fake_report)
    app.dependency_overrides[get_db] = lambda: mock_db

    with TestClient(app) as test_client:
        response = test_client.get(f"/api/events/{fake_event.id}/report")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "done"
    assert "narrative" in data
    assert "forecast" in data
    app.dependency_overrides.clear()


def test_get_report_returns_404_when_no_report_exists(client):
    app, get_db = client
    fake_event = _make_fake_event()
    mock_db = _make_mock_db(fake_event, None)
    app.dependency_overrides[get_db] = lambda: mock_db

    with TestClient(app) as test_client:
        response = test_client.get(f"/api/events/{fake_event.id}/report")

    assert response.status_code == 404
    app.dependency_overrides.clear()


def test_reporting_health_endpoint(client):
    app, get_db = client
    with TestClient(app) as test_client:
        response = test_client.get("/api/reporting/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "llm_enabled" in data
    assert "endpoints" in data

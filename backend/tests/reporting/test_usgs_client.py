"""
tests/reporting/test_usgs_client.py
-------------------------------------
Unit tests for the USGS enrichment client.
External HTTP is mocked — these tests never hit the real USGS API.
"""

import pytest
from unittest.mock import MagicMock, patch

from backend.reporting.usgs_client import fetch_and_cache, _fetch_from_usgs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_GEOJSON = {
    "type": "Feature",
    "properties": {
        "mag": 6.4,
        "place": "15 km NW of Ridgecrest, CA",
        "time": 1562008801040,   # milliseconds since epoch
        "felt": 97,
        "alert": "yellow",
        "detail": "https://earthquake.usgs.gov/earthquakes/eventpage/ci38443183",
    },
    "geometry": {
        "type": "Point",
        "coordinates": [-117.7428, 35.7695, 8.0],  # lon, lat, depth_km
    },
    "id": "ci38443183",
}


def _make_db(cached_row=None):
    """Return a mock SQLAlchemy session."""
    db = MagicMock()
    query_mock = MagicMock()
    query_mock.filter.return_value.first.return_value = cached_row
    db.query.return_value = query_mock
    return db


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_cached_row_without_network_call():
    """If a matching EventMetadata row exists in DB, no HTTP call is made."""
    from backend.models.database import EventMetadata

    fake_row = EventMetadata(
        usgs_id="ci38443183",
        magnitude=6.4,
        depth_km=8.0,
        lat=35.7695,
        lon=-117.7428,
        place="15 km NW of Ridgecrest, CA",
        felt=97,
        pager_alert="yellow",
        raw_json=SAMPLE_GEOJSON,
    )
    db = _make_db(cached_row=fake_row)

    with patch("backend.reporting.usgs_client._fetch_from_usgs") as mock_fetch:
        result = fetch_and_cache("ci38443183", db)
        mock_fetch.assert_not_called()

    assert result["magnitude"] == 6.4
    assert result["place"] == "15 km NW of Ridgecrest, CA"
    assert result["pager_alert"] == "yellow"
    assert result["felt"] == 97


def test_fetches_from_usgs_on_cache_miss():
    """On cache miss, the USGS API is called and result is cached."""
    db = _make_db(cached_row=None)

    with patch("backend.reporting.usgs_client._fetch_from_usgs", return_value=SAMPLE_GEOJSON):
        result = fetch_and_cache("ci38443183", db)

    assert result["magnitude"] == 6.4
    assert result["depth_km"] == 8.0
    assert result["lat"] == pytest.approx(35.7695)
    assert result["origin_time_iso"] is not None


def test_returns_empty_dict_when_usgs_unreachable():
    """When the USGS API call fails, an empty dict is returned (no crash)."""
    db = _make_db(cached_row=None)

    with patch("backend.reporting.usgs_client._fetch_from_usgs", return_value={}):
        result = fetch_and_cache("nonexistent-id", db)

    assert result == {}


def test_fetch_from_usgs_http_error():
    """_fetch_from_usgs handles HTTP 404 gracefully."""
    import requests
    from unittest.mock import patch, MagicMock

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")

    with patch("backend.reporting.usgs_client.requests.get", return_value=mock_resp):
        result = _fetch_from_usgs("bad-id")

    assert result == {}

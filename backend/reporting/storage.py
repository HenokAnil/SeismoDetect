"""
backend/reporting/storage.py
------------------------------
Upload report artefacts (JSON, HTML, PDF) to MinIO and return S3 keys.

Keeps storage concerns decoupled from generation logic.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from backend.api.minio_client import minio_storage

logger = logging.getLogger(__name__)

_REPORT_PREFIX = "reports"


def upload_report_artifacts(
    report_id: str,
    narrative: str,
    context: dict,
    forecast: dict,
) -> dict:
    """
    Upload all report artefacts to MinIO.

    Returns dict with keys: json_s3_key, html_s3_key, pdf_s3_key (may be None).
    """
    keys: dict = {}

    # --- JSON ---
    try:
        payload = {
            "report_id":  report_id,
            "narrative":  narrative,
            "forecast":   forecast,
            "similar":    context.get("similar", []),
            "current":    context.get("current", {}),
        }
        json_bytes = json.dumps(payload, indent=2, default=str).encode("utf-8")
        json_key   = f"{_REPORT_PREFIX}/{report_id}/report.json"
        minio_storage.upload_bytes(json_key, json_bytes, content_type="application/json")
        keys["json_s3_key"] = json_key
        logger.info("Uploaded report JSON to MinIO: %s", json_key)
    except Exception as exc:
        logger.error("Failed to upload report JSON: %s", exc)
        keys["json_s3_key"] = None

    # --- HTML ---
    try:
        html_str   = _render_html(report_id, narrative, context, forecast)
        html_bytes = html_str.encode("utf-8")
        html_key   = f"{_REPORT_PREFIX}/{report_id}/report.html"
        minio_storage.upload_bytes(html_key, html_bytes, content_type="text/html")
        keys["html_s3_key"] = html_key
        logger.info("Uploaded report HTML to MinIO: %s", html_key)
    except Exception as exc:
        logger.error("Failed to upload report HTML: %s", exc)
        keys["html_s3_key"] = None

    # --- PDF (optional, requires weasyprint) ---
    pdf_enabled = os.getenv("REPORT_PDF", "false").lower() == "true"
    if pdf_enabled:
        try:
            from weasyprint import HTML as WeasyHTML  # type: ignore
            pdf_bytes = WeasyHTML(string=html_str).write_pdf()
            pdf_key   = f"{_REPORT_PREFIX}/{report_id}/report.pdf"
            minio_storage.upload_bytes(pdf_key, pdf_bytes, content_type="application/pdf")
            keys["pdf_s3_key"] = pdf_key
            logger.info("Uploaded report PDF to MinIO: %s", pdf_key)
        except ImportError:
            logger.warning("weasyprint not installed — PDF export skipped")
            keys["pdf_s3_key"] = None
        except Exception as exc:
            logger.error("Failed to generate/upload PDF: %s", exc)
            keys["pdf_s3_key"] = None
    else:
        keys["pdf_s3_key"] = None

    return keys


# ---------------------------------------------------------------------------
# HTML renderer (Jinja2-free, pure f-string for zero extra deps)
# ---------------------------------------------------------------------------

def _render_html(
    report_id: str,
    narrative: str,
    context: dict,
    forecast: dict,
) -> str:
    """Render a clean standalone HTML page for the report."""
    cur     = context.get("current", {})
    similar = context.get("similar", [])
    fc      = forecast

    # Build similar events table rows
    sim_rows = ""
    for s in similar[:5]:
        pager_badge = ""
        if s.get("pager_alert"):
            colour_map = {"green": "#22c55e", "yellow": "#eab308",
                          "orange": "#f97316", "red": "#ef4444"}
            colour = colour_map.get(s["pager_alert"].lower(), "#94a3b8")
            pager_badge = (
                f'<span style="background:{colour};color:#fff;'
                f'padding:2px 8px;border-radius:4px;font-size:0.75rem;">'
                f'{s["pager_alert"].upper()}</span>'
            )
        usgs_href = (
            f'<a href="https://earthquake.usgs.gov/earthquakes/eventpage/{s["usgs_id"]}" '
            f'target="_blank">{s["usgs_id"]}</a>'
            if s.get("usgs_id") and s["usgs_id"] != "unknown"
            else s.get("usgs_id", "—")
        )
        sim_rows += f"""
        <tr>
          <td>{usgs_href} {pager_badge}</td>
          <td>{s.get('magnitude', '—')}</td>
          <td>{s.get('place', '—')}</td>
          <td>{s.get('origin_time', '—')}</td>
          <td><strong>{s['similarity']}</strong></td>
        </tr>"""

    # Convert markdown-ish narrative to simple paragraphs
    narrative_html = ""
    for line in narrative.splitlines():
        stripped = line.strip()
        if not stripped:
            narrative_html += "<br/>"
        elif stripped.startswith("## "):
            narrative_html += f"<h2>{stripped[3:]}</h2>"
        elif stripped.startswith("### "):
            narrative_html += f"<h3>{stripped[4:]}</h3>"
        elif stripped.startswith("- ") or stripped.startswith("* "):
            narrative_html += f"<li>{stripped[2:]}</li>"
        elif stripped.startswith("> "):
            narrative_html += f'<blockquote style="color:#dc2626;">{stripped[2:]}</blockquote>'
        elif stripped.startswith("---"):
            narrative_html += "<hr/>"
        else:
            narrative_html += f"<p>{stripped}</p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>SeismoDetect Report — Event {cur.get('event_id', '')[:8]}</title>
  <style>
    body {{font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           max-width: 900px; margin: 40px auto; color: #1e293b; line-height: 1.6;}}
    h1   {{color: #1e40af; border-bottom: 2px solid #3b82f6; padding-bottom: 8px;}}
    h2   {{color: #1d4ed8; margin-top: 2em;}}
    h3   {{color: #2563eb;}}
    table{{border-collapse: collapse; width: 100%; margin: 1em 0;}}
    th   {{background: #e0e7ff; padding: 8px 12px; text-align: left; font-size: 0.85rem;}}
    td   {{padding: 7px 12px; border-bottom: 1px solid #e2e8f0; font-size: 0.9rem;}}
    tr:hover td {{background: #f8fafc;}}
    .fact-card {{background: #f0f9ff; border-left: 4px solid #3b82f6;
                 padding: 12px 16px; border-radius: 4px; margin: 1.5em 0;}}
    .fact-card p {{margin: 4px 0; font-size: 0.9rem;}}
    .badge {{display:inline-block; background:#dbeafe; color:#1e40af;
             padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600;}}
    footer {{font-size: 0.75rem; color: #94a3b8; margin-top: 3em; border-top: 1px solid #e2e8f0; padding-top: 1em;}}
    blockquote {{border-left: 4px solid #dc2626; padding: 8px 12px;
                 background: #fef2f2; margin: 1em 0; border-radius: 0 4px 4px 0;}}
    hr {{border: none; border-top: 1px solid #e2e8f0; margin: 2em 0;}}
  </style>
</head>
<body>
  <h1>🌍 SeismoDetect — Seismic Event Report</h1>
  <p style="color:#64748b; font-size:0.9rem;">
    Report ID: <code>{report_id}</code> &nbsp;|&nbsp;
    Event: <code>{cur.get('event_id', '—')}</code>
  </p>

  <div class="fact-card">
    <h3 style="margin:0 0 8px;">⚡ Detected Event (Current)</h3>
    <p><strong>Station:</strong> {cur.get('station', '—')} ({cur.get('network', '—')})</p>
    <p><strong>Origin time:</strong> {cur.get('origin_time', '—')}</p>
    <p><strong>Magnitude proxy:</strong> <span class="badge">{cur.get('magnitude_proxy', '—')}</span></p>
    <p><strong>Detection confidence:</strong> {(cur.get('detection_prob') or 0) * 100:.1f} %</p>
    <p><strong>Depth:</strong> {cur.get('depth_km', 'unknown')} km</p>
  </div>

  <div class="fact-card" style="background:#f0fdf4; border-color:#22c55e;">
    <h3 style="margin:0 0 8px;">📊 Quantitative Forecast</h3>
    <p><strong>Method:</strong> {fc.get('method', '—')}</p>
    <p><strong>Predicted PGA (median):</strong> {fc.get('pga_g_median', '—')} g</p>
    <p><strong>PGA 95 % CI:</strong> [{fc.get('pga_g_lower', '?')}, {fc.get('pga_g_upper', '?')}] g</p>
    <p><strong>MMI estimate:</strong> <span class="badge">{fc.get('mmi', '—')}</span></p>
    <p><strong>Damage band:</strong> {fc.get('damage_band', '—')}</p>
    <p><strong>Confidence:</strong> {fc.get('confidence', '—')}</p>
  </div>

  <h2>🔍 Historical Similar Events (Top {len(similar[:5])})</h2>
  <table>
    <thead>
      <tr>
        <th>Event ID</th><th>Magnitude</th><th>Location</th>
        <th>Time (UTC)</th><th>Similarity</th>
      </tr>
    </thead>
    <tbody>{sim_rows}</tbody>
  </table>

  <h2>📝 Generated Narrative Report</h2>
  <div style="background:#f8fafc; padding:16px 20px; border-radius:6px; border:1px solid #e2e8f0;">
    {narrative_html}
  </div>

  <footer>
    <p>SeismoDetect Platform &bull; Automated RAG Report &bull;
       Generated with EQTransformer &amp; Qdrant &amp; T5</p>
    <p>⚠️ This report is auto-generated. All numeric forecasts are model estimates
       and must be verified against instrument data before emergency response.</p>
  </footer>
</body>
</html>"""

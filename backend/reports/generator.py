import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

def generate_pdf_report(event_data: dict, output_path: str) -> str:
    """
    Generates a professional seismic event report in PDF format using ReportLab.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a'),
        alignment=TA_CENTER
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#64748b'),
        alignment=TA_CENTER
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1e40af'),
        spaceBefore=12,
        spaceAfter=6
    )

    cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#1e293b')
    )

    cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#0f172a')
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph("SeismoDetect - Seismic Event Analysis Report", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Automated Deep Learning Phase Picking & Waveform Intelligence", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#3b82f6'), spaceAfter=15))

    # Event Overview Section
    story.append(Paragraph("Event Overview", section_heading))
    overview_data = [
        [Paragraph("Event ID", cell_bold), Paragraph(str(event_data.get("event_id", "--")), cell_style),
         Paragraph("Signal ID", cell_bold), Paragraph(str(event_data.get("signal_id", "--")), cell_style)],
        [Paragraph("Origin Time", cell_bold), Paragraph(str(event_data.get("origin_time", "--")), cell_style),
         Paragraph("Magnitude Proxy", cell_bold), Paragraph(str(event_data.get("magnitude_proxy", "--")), cell_style)],
        [Paragraph("Detection Confidence", cell_bold), Paragraph(f"{float(event_data.get('detection_prob', 0) or 0):.2%}", cell_style),
         Paragraph("Model Version", cell_bold), Paragraph(str(event_data.get("model_version", "EQTransformer v1")), cell_style)]
    ]
    overview_table = Table(overview_data, colWidths=[120, 150, 120, 150])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 12))

    # Signal & Station Details
    story.append(Paragraph("Station & Stream Metadata", section_heading))
    station_data = [
        [Paragraph("Network", cell_bold), Paragraph(str(event_data.get("network", "--")), cell_style),
         Paragraph("Station", cell_bold), Paragraph(str(event_data.get("station", "--")), cell_style)],
        [Paragraph("Channel", cell_bold), Paragraph(str(event_data.get("channel", "--")), cell_style),
         Paragraph("Sampling Rate", cell_bold), Paragraph(f"{event_data.get('sample_rate', '--')} Hz", cell_style)],
        [Paragraph("Start Time", cell_bold), Paragraph(str(event_data.get("starttime", "--")), cell_style),
         Paragraph("End Time", cell_bold), Paragraph(str(event_data.get("endtime", "--")), cell_style)]
    ]
    station_table = Table(station_data, colWidths=[120, 150, 120, 150])
    station_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(station_table)
    story.append(Spacer(1, 12))

    # Signal Quality Section
    story.append(Paragraph("Signal Quality & Noise Assessment", section_heading))
    quality = event_data.get("quality", {})
    quality_data = [
        [Paragraph("Avg SNR", cell_bold), Paragraph(f"{float(quality.get('snr', 0) or 0):.2f} dB", cell_style),
         Paragraph("Noise Classification", cell_bold), Paragraph(str(quality.get("noise_label", "clean")).upper(), cell_style)],
        [Paragraph("Quality Score", cell_bold), Paragraph(f"{float(quality.get('quality_score', 1.0) or 1.0):.2%}", cell_style),
         Paragraph("Clipping Detected", cell_bold), Paragraph("Yes" if quality.get("clipping_flag") else "No (Clean Digitizer)", cell_style)]
    ]
    quality_table = Table(quality_data, colWidths=[120, 150, 120, 150])
    quality_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(quality_table)
    story.append(Spacer(1, 12))

    # Phase Picks Section
    story.append(Paragraph("Detected Phase Arrivals (P & S Picks)", section_heading))
    picks = event_data.get("picks", [])
    picks_rows = [
        [Paragraph("Phase", cell_bold), Paragraph("Arrival Time (UTC)", cell_bold), Paragraph("Confidence", cell_bold), Paragraph("Uncertainty", cell_bold)]
    ]
    if picks:
        for p in picks:
            conf = float(p.get("confidence", 0) or 0)
            unc = p.get("uncertainty_s")
            unc_str = f"±{unc:.3f} s" if unc is not None else "N/A"
            picks_rows.append([
                Paragraph(f"<b>{p.get('phase', '--')} Phase</b>", cell_style),
                Paragraph(str(p.get("pick_time", "--")), cell_style),
                Paragraph(f"{conf:.1%}", cell_style),
                Paragraph(unc_str, cell_style)
            ])
    else:
        picks_rows.append([Paragraph("No distinct phase arrivals identified above detection threshold.", cell_style), Paragraph("", cell_style), Paragraph("", cell_style), Paragraph("", cell_style)])

    picks_table = Table(picks_rows, colWidths=[100, 220, 110, 110])
    picks_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e0e7ff')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#94a3b8')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(picks_table)
    story.append(Spacer(1, 12))

    # Similar Events Section
    similar = event_data.get("similar_events", [])
    if similar:
        story.append(Paragraph("Historical Similar Events (Qdrant 128-d Vector Match)", section_heading))
        sim_rows = [
            [Paragraph("Matched Event ID", cell_bold), Paragraph("Similarity Score", cell_bold), Paragraph("Notes", cell_bold)]
        ]
        for s in similar:
            score = float(s.get("score", 0) or 0)
            sim_rows.append([
                Paragraph(str(s.get("similar_event_id", "--")), cell_style),
                Paragraph(f"{score:.3f} ({score*100:.1f}%)", cell_style),
                Paragraph(str(s.get("time", "Historical Catalog")), cell_style)
            ])
        sim_table = Table(sim_rows, colWidths=[240, 140, 160])
        sim_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#94a3b8')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(sim_table)
        story.append(Spacer(1, 12))

    # Quantitative Forecast Section (from GMPE / RAG pipeline)
    forecast = event_data.get("forecast")
    if forecast:
        story.append(Paragraph("Quantitative Forecast (GMPE)", section_heading))
        forecast_data = [
            [Paragraph("Method", cell_bold),
             Paragraph(str(forecast.get("method", "--")), cell_style),
             Paragraph("Confidence", cell_bold),
             Paragraph(str(forecast.get("confidence", "--")).upper(), cell_style)],
            [Paragraph("Predicted PGA (median)", cell_bold),
             Paragraph(f"{forecast.get('pga_g_median', '--')} g", cell_style),
             Paragraph("PGA 95% CI", cell_bold),
             Paragraph(f"[{forecast.get('pga_g_lower', '?')}, {forecast.get('pga_g_upper', '?')}] g", cell_style)],
            [Paragraph("MMI Estimate", cell_bold),
             Paragraph(str(forecast.get("mmi", "--")), cell_style),
             Paragraph("Damage Band", cell_bold),
             Paragraph(str(forecast.get("damage_band", "--")), cell_style)],
        ]
        forecast_table = Table(forecast_data, colWidths=[120, 150, 120, 150])
        forecast_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f0fdf4')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#86efac')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1fae5')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(forecast_table)
        story.append(Spacer(1, 12))

    # AI Narrative Section (from RAG generator)
    narrative = event_data.get("narrative")
    if narrative:
        story.append(Paragraph("AI-Generated Narrative Report", section_heading))
        narrative_style = ParagraphStyle(
            'Narrative',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            leading=13,
            textColor=colors.HexColor('#1e293b'),
            spaceAfter=4,
        )
        # Render narrative line by line — map markdown headings to bold paragraphs
        for line in narrative.splitlines():
            stripped = line.strip()
            if not stripped:
                story.append(Spacer(1, 4))
            elif stripped.startswith("## ") or stripped.startswith("### "):
                hdr = stripped.lstrip("#").strip()
                story.append(Paragraph(f"<b>{hdr}</b>", narrative_style))
            elif stripped.startswith("---"):
                story.append(HRFlowable(width="100%", thickness=0.5,
                                        color=colors.HexColor('#cbd5e1'),
                                        spaceBefore=6, spaceAfter=6))
            else:
                story.append(Paragraph(stripped, narrative_style))
        story.append(Spacer(1, 12))

    # Footer note
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cbd5e1'), spaceBefore=20, spaceAfter=8))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#94a3b8'),
        alignment=TA_CENTER
    )
    story.append(Paragraph("SeismoDetect Platform • Automated Event Report • Generated with PyTorch & SeisBench & Qdrant", footer_style))

    doc.build(story)
    return output_path


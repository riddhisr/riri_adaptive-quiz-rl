# app/reporting.py
"""
Robust PDF report generator using ReportLab with a safe FPDF fallback.
This avoids FPDF word-wrapping problems and handles Unicode/wrapping better.
"""

import os
import json
import re
import textwrap
from pathlib import Path

# Try to import reportlab; if unavailable we fall back to the safe FPDF routine
try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# fpdf fallback
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SESSION_JSON_DIR = DATA_DIR / "session_details"
REPORTS_DIR = DATA_DIR / "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# -----------------------------
# Text sanitization helpers
# -----------------------------
def sanitize_text(s):
    """Keep Unicode but remove control characters; collapse spaces."""
    if s is None:
        return ""
    s = str(s)
    # remove control chars except newline/tab
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]+", " ", s)
    # replace long dashes with simple hyphen
    s = s.replace("—", "-").replace("–", "-")
    # collapse multiple whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def break_long_tokens(s, maxlen=40):
    """
    Insert spaces into extremely long tokens (no whitespace) so libraries can wrap them.
    """
    def repl(m):
        tok = m.group(0)
        parts = [tok[i:i+maxlen] for i in range(0, len(tok), maxlen)]
        return " ".join(parts)
    return re.sub(r'\S{' + str(maxlen+1) + r',}', repl, s)

def safe_text(s):
    s = sanitize_text(s)
    s = break_long_tokens(s, maxlen=40)
    return s

# -----------------------------
# small util: save cumulative reward plot
# -----------------------------
def save_cumulative_plot(df, out_path):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(df["q_no"], df["reward"].cumsum(), marker="o")
    ax.set_title("Cumulative reward")
    ax.set_xlabel("Question")
    ax.set_ylabel("Cumulative reward")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# -----------------------------
# ReportLab implementation
# -----------------------------
def _generate_report_reportlab(session_id, out_pdf_path):
    jpath = SESSION_JSON_DIR / f"{session_id}.json"
    if not jpath.exists():
        raise FileNotFoundError("Session JSON not found: " + str(jpath))
    sess = json.loads(jpath.read_text(encoding="utf-8"))
    interactions = sess.get("interactions", [])
    df = pd.DataFrame(interactions)
    if df.empty:
        raise ValueError("No interactions in session.")

    accuracy = df["correct"].mean()
    avg_time = df["time_taken"].mean()
    total_reward = df["reward"].sum()

    out_pdf_path = Path(out_pdf_path)
    out_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # prepare image
    tmp_img = REPORTS_DIR / f"{session_id}_reward.png"
    save_cumulative_plot(df, str(tmp_img))

    # build document
    doc = SimpleDocTemplate(str(out_pdf_path), pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    normal.fontSize = 10
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], spaceAfter=6)

    flowables = []
    title = f"Adaptive Quiz Report - Session {session_id}"
    flowables.append(Paragraph(safe_text(title), h1))
    flowables.append(Spacer(1, 6))

    # summary table
    summary_data = [
        ["Accuracy", f"{accuracy*100:.2f}%"],
        ["Avg time per Q (s)", f"{avg_time:.2f}"],
        ["Total reward", f"{total_reward:.2f}"],
        ["Questions recorded", f"{len(df)}"]
    ]
    tbl = Table(summary_data, colWidths=[90*mm, 60*mm])
    tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(0,-1), colors.lightgrey),
                             ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                             ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
                             ("BOX", (0,0), (-1,-1), 0.25, colors.grey)]))
    flowables.append(tbl)
    flowables.append(Spacer(1, 8))

    # insert plot (fit to width)
    try:
        img = RLImage(str(tmp_img))
        max_width = doc.width
        if img.drawWidth > max_width:
            ratio = max_width / img.drawWidth
            img.drawWidth = max_width
            img.drawHeight = img.drawHeight * ratio
        flowables.append(img)
    except Exception:
        # ignore image errors
        pass

    flowables.append(Spacer(1, 8))
    flowables.append(Paragraph("Interactions (first 40):", h2))

    # interactions table: create safe text and limit rows
    rows = [["Q#", "Difficulty", "Chosen", "Correct", "Time(s)", "Reward", "Question (truncated)"]]
    for idx, row in df.head(40).iterrows():
        qno = int(row.get("q_no", idx+1))
        diff = safe_text(row.get("difficulty",""))
        chosen = safe_text(row.get("chosen",""))
        correct = int(row.get("correct",0))
        t = float(row.get("time_taken",0.0))
        r = float(row.get("reward",0.0))
        qtxt = safe_text(row.get("question",""))[:200]  # truncate question to 200 chars
        rows.append([str(qno), diff, chosen, str(correct), f"{t:.2f}", f"{r:.2f}", qtxt])

    t = Table(rows, colWidths=[12*mm, 22*mm, 30*mm, 10*mm, 16*mm, 16*mm, doc.width - (12+22+30+10+16+16)*mm])
    t.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),
                           ("ALIGN",(3,0),(5,-1),"CENTER"),
                           ("GRID",(0,0),(-1,-1),0.25,colors.grey),
                           ("BACKGROUND",(0,0),(-1,0),colors.lightblue)]))
    flowables.append(t)
    flowables.append(PageBreak())
    # footer page
    flowables.append(Paragraph(safe_text(f"Generated by Adaptive QuizRL+ — session: {session_id}"), normal))

    doc.build(flowables)
    return str(out_pdf_path)

# -----------------------------
# FPDF fallback (kept defensive)
# -----------------------------
def _generate_report_fpdf(session_id, out_pdf_path):
    jpath = SESSION_JSON_DIR / f"{session_id}.json"
    if not jpath.exists():
        raise FileNotFoundError("Session JSON not found: " + str(jpath))
    sess = json.loads(jpath.read_text(encoding="utf-8"))
    interactions = sess.get("interactions", [])
    df = pd.DataFrame(interactions)
    if df.empty:
        raise ValueError("No interactions in session.")

    accuracy = df["correct"].mean()
    avg_time = df["time_taken"].mean()
    total_reward = df["reward"].sum()

    out_pdf_path = Path(out_pdf_path)
    out_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # small plot
    tmp_img = REPORTS_DIR / f"{session_id}_reward.png"
    save_cumulative_plot(df, str(tmp_img))

    # Use small font sizes and robust breaking
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    title = f"Adaptive Quiz Report - Session {session_id}"
    pdf.multi_cell(0, 7, break_long_tokens(sanitize_text(title), maxlen=60))
    pdf.ln(2)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, f"Accuracy: {accuracy*100:.2f}%")
    pdf.multi_cell(0, 6, f"Avg time per Q: {avg_time:.2f}s")
    pdf.multi_cell(0, 6, f"Total reward: {total_reward:.2f}")
    pdf.ln(4)
    # image fit
    page_w = pdf.w - 2*pdf.l_margin
    try:
        pdf.image(str(tmp_img), x=pdf.l_margin, w=page_w)
    except Exception:
        pass
    pdf.ln(4)
    pdf.set_font("Courier", size=8)
    for idx, row in df.head(40).iterrows():
        qno = int(row.get("q_no", idx+1))
        diff = sanitize_text(row.get("difficulty",""))
        chosen = sanitize_text(row.get("chosen",""))
        correct = int(row.get("correct",0))
        t = float(row.get("time_taken",0.0))
        r = float(row.get("reward",0.0))
        qtxt = break_long_tokens(sanitize_text(row.get("question","")), maxlen=60)[:400]
        line = f"Q{qno} | {diff} | chosen:{chosen} | c:{correct} | t:{t:.2f}s | r:{r:.2f}"
        pdf.multi_cell(0, 5, line)
        pdf.set_x(pdf.l_margin + 6)
        pdf.multi_cell(0, 5, qtxt)
    pdf.output(str(out_pdf_path))
    return str(out_pdf_path)

# -----------------------------
# Public function
# -----------------------------
def generate_session_report(session_id, out_pdf_path=None):
    if out_pdf_path is None:
        out_pdf_path = REPORTS_DIR / f"report_{session_id}.pdf"
    if REPORTLAB_AVAILABLE:
        return _generate_report_reportlab(session_id, out_pdf_path)
    elif FPDF_AVAILABLE:
        # fallback
        return _generate_report_fpdf(session_id, out_pdf_path)
    else:
        raise RuntimeError("No PDF library available. Install 'reportlab' or 'fpdf'.")


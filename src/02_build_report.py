"""
Step 2 - Compile all charts + insight captions into one PDF visual report.
Run AFTER 01_analysis_and_charts.py:   python src/02_build_report.py
"""
import json
from pathlib import Path

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)


STUDENT_NAME = "ALOK GUPTA"
ADMISSION_NUMBER = "2026M01410061", "Sec(A)"
COLLEGE = "ABES Engineering College"


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
CHART_DIR = OUT / "charts"
REPORT_PDF = ROOT / "report" / "Retail_Sales_Visual_Report.pdf"
REPORT_PDF.parent.mkdir(exist_ok=True)

meta = json.loads((OUT / "charts_meta.json").read_text())
KPI, LOG, CHARTS = meta["kpi"], meta["cleaning"], meta["charts"]
df = pd.read_csv(ROOT / "data" / "cleaned_retail_data.csv")

# DejaVu fonts ship with matplotlib and support the rupee sign (₹)
font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(font_dir / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", str(font_dir / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV", boldItalic="DV-B")

NAVY, ORANGE, GREY = colors.HexColor("#1F4E79"), colors.HexColor("#E07B39"), colors.HexColor("#555555")
S = {
    "title": ParagraphStyle("title", fontName="DV-B", fontSize=26, leading=32, textColor=NAVY, alignment=TA_CENTER),
    "sub": ParagraphStyle("sub", fontName="DV", fontSize=13, leading=18, textColor=GREY, alignment=TA_CENTER),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=16, leading=20, textColor=NAVY, spaceBefore=6, spaceAfter=8, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=12.5, leading=16, textColor=NAVY, spaceBefore=4, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="DV", fontSize=9.5, leading=14),
    "cap": ParagraphStyle("cap", fontName="DV", fontSize=9.5, leading=14, textColor=colors.HexColor("#222222"),
        backColor=colors.HexColor("#EEF3F8"), borderPadding=(6, 8, 6, 8), spaceBefore=6, spaceAfter=16),
}


def m(x):
    return f"₹{x / 1e6:.1f}M"


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(2 * cm, 1.2 * cm, "Retail Sales Visual Report")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


story = []

# ---------------- cover page ----------------
story += [Spacer(1, 5 * cm), Paragraph("Turning Retail Sales Data<br/>into a Visual Story", S["title"]), Spacer(1, 0.6 * cm),
          Paragraph("Data Visualization Mini Project – Data Analyst Training Program", S["sub"]), Spacer(1, 0.4 * cm),
          Paragraph("Retail transactions · 2023–2024 · four regions of India", S["sub"]), Spacer(1, 4 * cm),
          Paragraph(f"<b>Prepared By: {STUDENT_NAME}</b><br/>"f"Admission Number: {ADMISSION_NUMBER}<br/>"f"{COLLEGE}",S["sub"]),PageBreak()
        ]

# ---------------- executive summary ----------------
region = df.groupby("Region")["Sales"].sum()
cat = df.groupby("Product_Category")["Sales"].sum()
story.append(Paragraph("Executive Summary", S["h1"]))
kpi_rows = [["Total Sales", "Total Profit", "Profit Margin", "Avg Order Value", "Return Rate"],
            [m(KPI["total_sales"]), m(KPI["total_profit"]), f"{KPI['profit_margin']:.1%}", f"₹{KPI['avg_order_value']:,.0f}", f"{KPI['return_rate']:.1%}"]]
t = Table(kpi_rows, colWidths=[3.4 * cm] * 5)
t.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, 0), "DV"), ("FONTSIZE", (0, 0), (-1, 0), 8), ("TEXTCOLOR", (0, 0), (-1, 0), GREY),
    ("FONTNAME", (0, 1), (-1, 1), "DV-B"), ("FONTSIZE", (0, 1), (-1, 1), 13), ("TEXTCOLOR", (0, 1), (-1, 1), NAVY),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EEF3F8")),
    ("BOX", (0, 0), (-1, -1), 0.6, colors.white), ("INNERGRID", (0, 0), (-1, -1), 2, colors.white),
    ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
story += [t, Spacer(1, 0.5 * cm)]

story.append(Paragraph("Key takeaways", S["h2"]))
takeaways = [
    f"Sales are flat year on year; monthly revenue swings between roughly {m(df.groupby('Month')['Sales'].sum().min())} and {m(df.groupby('Month')['Sales'].sum().max())}.",
    f"{cat.idxmax()} generates {cat.max() / cat.sum():.0%} of all sales, so category mix drives the business more than geography.",
    f"{region.idxmax()} is the strongest region ({m(region.max())}); {region.idxmin()} is the weakest ({m(region.min())}).",
    f"Profit follows sales closely (margin about {KPI['profit_margin']:.0%}); customer age and income have almost no effect on order value.",
    f"About {KPI['return_rate']:.0%} of orders are returned, and satisfaction does not change much with delivery speed.",
]
for tk in takeaways:
    story.append(Paragraph("•&nbsp;&nbsp;" + tk, S["body"]))
    story.append(Spacer(1, 3))

story += [Spacer(1, 0.4 * cm), Paragraph("Data preparation (what was cleaned before charting)", S["h2"])]
clean_rows = [["Issue found in the raw file", "Action taken", "Rows affected"],
    ["Mixed date formats (4 styles)", "Parsed each style explicitly", f"all {LOG['rows_raw']:,}"],
    ["Inconsistent Region / Gender labels", "Mapped to standard names (e.g. 'Sth' → South)", "~250"],
    ["Duplicate records", "Removed duplicate rows", f"{LOG['exact_duplicate_rows_removed']}"],
    ["Impossible Age values (<18 or >100)", "Set to missing", f"{LOG['invalid_age_set_to_nan']}"],
    ["Invalid Discount (outside 0–50%)", "Set to missing", f"{LOG['invalid_discount_set_to_nan']}"],
    ["Extreme Income outliers (> Q3 + 3×IQR)", "Set to missing", f"{LOG['extreme_income_set_to_nan']}"],
    ["Delivery date before order date", "Delivery date set to missing", f"{LOG['delivery_before_order_set_to_nan']}"],
    ["Final dataset", "", f"{LOG['rows_clean']:,} rows"]]
ct = Table(clean_rows, colWidths=[6.6 * cm, 7 * cm, 3.3 * cm])
ct.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, -1), "DV"), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("FONTNAME", (0, 0), (-1, 0), "DV-B"), ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, -1), (-1, -1), "DV-B"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FA")]),
    ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#D5DBE3")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
story += [ct, PageBreak()]

# ---------------- charts in narrative order ----------------
current = None
for ch in CHARTS:
    block = []
    if ch["section"] != current:
        current = ch["section"]
        block.append(Paragraph(current, S["h1"]))
    img_path = CHART_DIR / ch["file"]
    iw, ih = __import__("PIL.Image", fromlist=["Image"]).open(img_path).size
    width = 15.5 * cm if iw / ih < 1.6 else 16.5 * cm
    height = width * ih / iw
    max_h = 9.6 * cm
    if height > max_h:
        width, height = width * max_h / height, max_h
    block += [Image(str(img_path), width=width, height=height),
        Paragraph(f"<b>Insight:</b> {ch['caption']}", S["cap"])]
    story.append(KeepTogether(block))

doc = SimpleDocTemplate(str(REPORT_PDF), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=2 * cm,
                        title="Retail Sales Visual Report", author=STUDENT_NAME)
doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=footer)
print("Report saved:", REPORT_PDF)

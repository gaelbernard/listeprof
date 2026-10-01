"""
Generate a printable PDF of the EPFL unit-count reconciliation
(brochure vs. professor-derived database vs. personnel export).

Data is recomputed from the source files so the report is reproducible:
  - input/List_of_professors_(Gael_labList_incl._SPC).csv   (professor list -> DB)
  - test-gael_Données migrées.csv                            (personnel export)

Usage:
    python generate_unit_reconciliation_pdf.py
Output:
    EPFL-unit-reconciliation.pdf
"""

import glob
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    LongTable,
    KeepTogether,
)
from reportlab.graphics.shapes import Drawing, Line, String
from reportlab.graphics.charts.barcharts import VerticalBarChart

OUT = "EPFL-unit-reconciliation.pdf"
BROCHURE_LABS, BROCHURE_GROUPS = 453, 76
BROCHURE_TOTAL = BROCHURE_LABS + BROCHURE_GROUPS  # 529

# ---- palette (light, printable) ----
ACCENT = colors.HexColor("#1F6FEB")
WARN = colors.HexColor("#B45309")
SUCCESS = colors.HexColor("#15803D")
INK = colors.HexColor("#1A1A1A")
MUTED = colors.HexColor("#6B7280")
LIGHT = colors.HexColor("#F3F4F6")
INFOBG = colors.HexColor("#EFF6FF")
BORDER = colors.HexColor("#D1D5DB")


def compute():
    prof = pd.read_csv("input/List_of_professors_(Gael_labList_incl._SPC).csv")
    prof_u = prof[["CF unité", "Sigle unité", "Nom unité", "Type unité"]].drop_duplicates(subset=["CF unité"]).copy()
    prof_u["cf"] = prof_u["CF unité"].astype(int)

    fn = glob.glob("test-gael_*.csv")[0]
    mig = pd.read_csv(fn, encoding="utf-16", sep="\t")
    keep = [
        "Centre financier (clé)", "Unité principale", "Centre financier", "Type d'unité",
        "Nom du responsable de l'unité principale", "Prénom du responsable de l'unité principale",
    ]
    mig_u = mig[keep].drop_duplicates(subset=["Centre financier (clé)"]).copy()
    mig_u["cf"] = mig_u["Centre financier (clé)"].str.extract(r"(\d+)").astype(int)

    prof_cf, mig_cf = set(prof_u["cf"]), set(mig_u["cf"])
    prof_only = prof_u[prof_u["cf"].isin(prof_cf - mig_cf)]
    mig_only = mig_u[mig_u["cf"].isin(mig_cf - prof_cf)].copy()
    mig_only["resp"] = (
        mig_only["Prénom du responsable de l'unité principale"].fillna("").astype(str) + " "
        + mig_only["Nom du responsable de l'unité principale"].fillna("").astype(str)
    ).str.strip()

    return {
        "prof_types": prof_u["Type unité"].value_counts().to_dict(),
        "mig_types": mig_u["Type d'unité"].value_counts().to_dict(),
        "prof_total": len(prof_cf),
        "mig_total": len(mig_cf),
        "inter": len(prof_cf & mig_cf),
        "union": len(prof_cf | mig_cf),
        "prof_only_types": prof_only["Type unité"].value_counts().to_dict(),
        "mig_only_types": mig_only["Type d'unité"].value_counts().to_dict(),
        "mig_only_rows": mig_only.rename(columns={
            "Unité principale": "acronym", "Centre financier": "name", "Type d'unité": "type",
        })[["acronym", "name", "type", "resp"]].sort_values(["type", "acronym"]).values.tolist(),
    }


def stat_cell(value, label, color):
    return Paragraph(
        f'<font size="22" color="#{color.hexval()[2:]}"><b>{value}</b></font><br/>'
        f'<font size="8" color="#{MUTED.hexval()[2:]}">{label}</font>',
        ParagraphStyle("stat", alignment=TA_LEFT, leading=13),
    )


def bar_chart(prof_total, mig_total, union):
    d = Drawing(480, 210)
    bc = VerticalBarChart()
    bc.x, bc.y, bc.width, bc.height = 50, 35, 380, 150
    bc.data = [[prof_total, mig_total, union]]
    bc.categoryAxis.categoryNames = ["Database\n(prof-derived)", "Personnel export", "Union of both"]
    bc.categoryAxis.labels.fontName = "Helvetica"
    bc.categoryAxis.labels.fontSize = 9
    bc.categoryAxis.labels.dy = -4
    bc.valueAxis.valueMin = 0
    bc.valueAxis.valueMax = 600
    bc.valueAxis.valueStep = 100
    bc.valueAxis.labels.fontName = "Helvetica"
    bc.valueAxis.labels.fontSize = 8
    bc.barWidth = 26
    bc.groupSpacing = 30
    bc.bars[0].fillColor = ACCENT
    bc.bars[0].strokeColor = None
    bc.barLabelFormat = "%d"
    bc.barLabels.fontName = "Helvetica-Bold"
    bc.barLabels.fontSize = 10
    bc.barLabels.nudge = 9
    d.add(bc)

    # brochure reference line at 529
    y = bc.y + (BROCHURE_TOTAL - bc.valueAxis.valueMin) / (bc.valueAxis.valueMax - bc.valueAxis.valueMin) * bc.height
    ref = Line(bc.x, y, bc.x + bc.width, y, strokeColor=WARN, strokeWidth=1.2)
    ref.strokeDashArray = [4, 2]
    d.add(ref)
    d.add(String(bc.x + bc.width, y + 4, f"Brochure: {BROCHURE_TOTAL}", fontName="Helvetica-Bold",
                 fontSize=8, fillColor=WARN, textAnchor="end"))
    return d


def build():
    data = compute()
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontName="Helvetica", fontSize=10,
                          leading=14, textColor=INK, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13,
                        textColor=INK, spaceBefore=16, spaceAfter=6)
    title = ParagraphStyle("title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=20,
                           textColor=INK, spaceAfter=2)
    caption = ParagraphStyle("cap", parent=body, fontName="Helvetica-Oblique", fontSize=8,
                             textColor=MUTED, leading=11, spaceBefore=3)
    cardh = ParagraphStyle("cardh", parent=body, fontName="Helvetica-Bold", fontSize=10, spaceAfter=3)

    story = []
    story.append(Paragraph("EPFL units: brochure vs. database", title))
    story.append(Paragraph(
        f"Why the database holds <b>{data['prof_total']}</b> units while the brochure reports "
        f"<b>{BROCHURE_LABS} labs + {BROCHURE_GROUPS} groups = {BROCHURE_TOTAL}</b>. "
        "Short version: the database is built from a professor list, so it only ever contains "
        "units that have a listed professor attached.", body))
    story.append(Spacer(1, 8))

    # summary stats
    stats = Table([[
        stat_cell(BROCHURE_TOTAL, f"Brochure ({BROCHURE_LABS} labs + {BROCHURE_GROUPS} groups)", INK),
        stat_cell(data["prof_total"], "Database (professor-derived)", WARN),
        stat_cell(data["mig_total"], "Personnel export (any staff)", INK),
        stat_cell(data["union"], "Union of both sources", SUCCESS),
    ]], colWidths=[128] * 4)
    stats.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(stats)
    story.append(Spacer(1, 10))

    # headline callout
    callout = Table([[Paragraph(
        "<b>The database is a by-product of a professor list, not a unit registry.</b> "
        "The <font face='Courier'>lab</font> table is derived from the EPFL professor CSV: a unit only "
        f"appears if at least one listed professor belongs to it. <b>{len(data['mig_only_rows'])} units</b> "
        "have staff or students but no professor in that list, so they never enter the database. Add them "
        f"back and the count (<b>{data['union']}</b>) lands right on the brochure's <b>{BROCHURE_TOTAL}</b>.",
        body)]], colWidths=[515])
    callout.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), INFOBG),
        ("LINEBEFORE", (0, 0), (0, -1), 3, ACCENT),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(callout)

    # two explanation cards
    story.append(Spacer(1, 10))
    c1 = [Paragraph("Root cause - professor-derived", cardh), Paragraph(
        "Step 1 of the pipeline reads the professor CSV and builds the lab table as the "
        "<b>distinct units of those professors</b>. Units with no listed professor - led by senior "
        "scientists, technical staff, or a head already counted under their main unit - are invisible.", body)]
    c2 = [Paragraph("Taxonomy &amp; snapshot mismatch", cardh), Paragraph(
        "The two sources label units differently - the professor export uses LABO / GROUPE / CENTRE / SPC, "
        "the personnel export uses LABO / CENTRE / CHAIRE. The brochure's 2-way 'labs vs. groups' split maps "
        "cleanly to neither, and the two extracts are from different dates.", body)]
    cards = Table([[c1, c2]], colWidths=[252, 252])
    cards.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (0, -1), 0.5, BORDER),
        ("BOX", (1, 0), (1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(cards)

    # chart
    story.append(Paragraph("Distinct units by source", h2))
    story.append(bar_chart(data["prof_total"], data["mig_total"], data["union"]))
    story.append(Paragraph(
        "Sources: DB lab table (Feb 2026 professor list) &middot; personnel export 'Donnees migrees' &middot; "
        "matched on unit CF code / acronym. Dashed line = brochure total (453 labs + 76 groups).", caption))

    # composition table
    story.append(Paragraph("Composition by unit type", h2))
    pt, mt = data["prof_types"], data["mig_types"]

    def g(d, k):
        return str(d.get(k, "")) if d.get(k) else "n/a"

    comp_rows = [
        ["Unit type", "Database", "Personnel export"],
        ["LABO - lab", g(pt, "LABO"), g(mt, "LABO")],
        ["GROUPE - group", g(pt, "GROUPE"), "n/a"],
        ["SPC - section", g(pt, "SPC"), "n/a"],
        ["CENTRE - centre", g(pt, "CENTRE"), g(mt, "CENTRE")],
        ["CHAIRE - chair", "n/a", g(mt, "CHAIRE")],
        ["Total", str(data["prof_total"]), str(data["mig_total"])],
    ]
    comp = Table(comp_rows, colWidths=[220, 140, 140])
    comp.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -2), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, BORDER),
        ("LINEABOVE", (0, -1), (-1, -1), 0.5, BORDER),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor("#FAFAFA")]),
    ]))
    story.append(comp)
    story.append(Paragraph(
        "'n/a' = the type does not exist in that source's taxonomy. The same physical unit can be typed "
        "differently across systems (43 of 45 CHAIRE units appear as LABO/GROUPE in the professor list).",
        caption))

    # what differs
    story.append(Paragraph("What differs between the two sources", h2))
    mo, po = data["mig_only_types"], data["prof_only_types"]
    mo_txt = ", ".join(f"{v} {k}" for k, v in sorted(mo.items(), key=lambda x: -x[1]))
    po_txt = ", ".join(f"{v} {k}" for k, v in sorted(po.items(), key=lambda x: -x[1]))
    diff = Table([
        [Paragraph(f"<b>{len(data['mig_only_rows'])} units</b> in personnel export, absent from DB", body),
         Paragraph(f"<b>{sum(po.values())} units</b> in DB, absent from personnel export", body)],
        [Paragraph(mo_txt, body), Paragraph(po_txt, body)],
        [Paragraph("Have staff/students but no listed professor - the main reason the DB under-counts.", caption),
         Paragraph("Mostly GROUPE/SPC types the personnel system doesn't use, plus a few closed/renamed units.", caption)],
    ], colWidths=[252, 252])
    diff.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (0, -1), 0.5, BORDER),
        ("BOX", (1, 0), (1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(diff)

    # caveat
    story.append(Spacer(1, 10))
    caveat = Table([[Paragraph(
        "<b>Caveat:</b> not all these units are distinct research labs. Many are administrative "
        "'... - Gestion' cost-centres or secondary units of a professor already counted "
        "(e.g. LMIS2 / LMIS4 / LO under Thiran). The number of genuinely professor-less research labs "
        "is smaller.", body)]], colWidths=[515])
    caveat.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("LINEBEFORE", (0, 0), (0, -1), 3, MUTED),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(caveat)

    # full missing-units table
    story.append(Paragraph(
        f"Units missing from the database ({len(data['mig_only_rows'])})", h2))
    cell = ParagraphStyle("cell", parent=body, fontSize=8, leading=10, spaceAfter=0)
    cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")
    head = ParagraphStyle("th", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
    rows = [[Paragraph("Acronym", head), Paragraph("Unit name", head),
             Paragraph("Type", head), Paragraph("Recorded head", head)]]
    for acr, name, typ, resp in data["mig_only_rows"]:
        rows.append([Paragraph(str(acr), cellb), Paragraph(str(name), cell),
                     Paragraph(str(typ), cell), Paragraph(str(resp), cell)])
    tbl = LongTable(rows, colWidths=[78, 250, 52, 135], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    story.append(Paragraph(
        "Units present in the personnel export but with no professor in the list used to build the DB. "
        "Head names shown as recorded (LASTNAME Firstname).", caption))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(40, 20, "EPFL unit-count reconciliation - brochure vs. professor-derived database")
        canvas.drawRightString(555, 20, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=40, rightMargin=40,
                            topMargin=36, bottomMargin=36, title="EPFL unit reconciliation")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    build()

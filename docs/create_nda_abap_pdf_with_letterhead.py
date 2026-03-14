#!/usr/bin/env python3
"""
Create ABAP AI Toolkit NDA PDF using only the TOP and BOTTOM letterhead sections.
Jayamma (on behalf of Paafekt) is the Disclosing Party creating and requesting others to sign.
Output: docs/NDA_ABAP_AI_Toolkit_Jayamma.pdf
"""
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
LETTERHEAD_TOP = DOCS_DIR / "letterhead" / "top.png"
LETTERHEAD_BOTTOM = DOCS_DIR / "letterhead" / "bottom.png"
OUTPUT_PDF = DOCS_DIR / "NDA_ABAP_AI_Toolkit_Jayamma.pdf"

TOP_BAND_HEIGHT = 120
BOTTOM_BAND_HEIGHT = 80


def add_letterhead_bands(canvas, doc):
    """Draw top and bottom letterhead images on the current page."""
    from reportlab.lib.utils import ImageReader

    page_width = doc.pagesize[0]
    page_height = doc.pagesize[1]
    if LETTERHEAD_TOP.exists():
        img = ImageReader(str(LETTERHEAD_TOP))
        iw, ih = img.getSize()
        scale = min((page_width - 72) / iw, TOP_BAND_HEIGHT / ih)
        w, h = iw * scale, ih * scale
        canvas.drawImage(str(LETTERHEAD_TOP), 36, page_height - h - 18, width=w, height=h)
    if LETTERHEAD_BOTTOM.exists():
        img = ImageReader(str(LETTERHEAD_BOTTOM))
        iw, ih = img.getSize()
        scale = min((page_width - 72) / iw, BOTTOM_BAND_HEIGHT / ih)
        w, h = iw * scale, ih * scale
        canvas.drawImage(str(LETTERHEAD_BOTTOM), 36, 12, width=w, height=h)


def create_nda_abap_pages():
    """Generate ABAP AI Toolkit NDA content as PDF pages using ReportLab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from io import BytesIO

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=TOP_BAND_HEIGHT + 36,
        bottomMargin=BOTTOM_BAND_HEIGHT + 36,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="NDATitle", fontSize=14, spaceAfter=12, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="NDAHeading", fontSize=11, spaceAfter=6, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="NDABody", fontSize=10, spaceAfter=6))

    story = []

    story.append(Paragraph("NON-DISCLOSURE AGREEMENT", styles["NDATitle"]))
    story.append(Paragraph("ABAP AI Toolkit", styles["NDAHeading"]))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("<b>Effective Date:</b> [DATE]", styles["NDABody"]))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph(
        "This Non-Disclosure Agreement (\"<b>Agreement</b>\") is entered into as of the Effective Date by and between:",
        styles["NDABody"],
    ))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("<b>Disclosing Party:</b>", styles["NDABody"]))
    story.append(Paragraph("Paafekt", styles["NDABody"]))
    story.append(Paragraph("[Full legal entity name and address] (\"<b>Discloser</b>\")", styles["NDABody"]))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("<b>Receiving Party:</b>", styles["NDABody"]))
    story.append(Paragraph("[FULL LEGAL NAME / COMPANY NAME]", styles["NDABody"]))
    story.append(Paragraph("[Address] (\"<b>Recipient</b>\")", styles["NDABody"]))
    story.append(Paragraph("Discloser and Recipient are each a \"<b>Party</b>\" and together the \"<b>Parties</b>.\"", styles["NDABody"]))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("1. Purpose and Scope", styles["NDAHeading"]))
    story.append(Paragraph(
        "Recipient may receive Confidential Information from Discloser in connection with <b>ABAP AI Toolkit</b> (the \"<b>Project</b>\"), "
        "including but not limited to related business or technical information.",
        styles["NDABody"],
    ))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("2. Definition of Confidential Information", styles["NDAHeading"]))
    story.append(Paragraph(
        "\"<b>Confidential Information</b>\" means any non-public information disclosed by Discloser to Recipient, orally, in writing, or by inspection, "
        "that relates to the Project (ABAP AI Toolkit), including without limitation: technical data, designs, specifications, and documentation; "
        "software, code, models, and configurations; business plans, pricing, and strategies; customer or partner information; "
        "any information marked or reasonably understood to be confidential.",
        styles["NDABody"],
    ))
    story.append(Paragraph(
        "Confidential Information does <b>not</b> include information that: (a) is or becomes publicly available through no fault of Recipient; "
        "(b) was rightfully known to Recipient without restriction before disclosure; (c) is independently developed by Recipient without use of Confidential Information; or "
        "(d) is rightfully received from a third party without restriction and without breach of any obligation of confidentiality.",
        styles["NDABody"],
    ))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("3. Obligations of Recipient", styles["NDAHeading"]))
    story.append(Paragraph("Recipient agrees to:", styles["NDABody"]))
    story.append(Paragraph(
        "• Hold all Confidential Information in strict confidence and not disclose it to any third party without Discloser's prior written consent.",
        styles["NDABody"],
    ))
    story.append(Paragraph(
        "• Use Confidential Information solely for the purpose of evaluating or collaborating on the Project (ABAP AI Toolkit) as authorized by Discloser.",
        styles["NDABody"],
    ))
    story.append(Paragraph(
        "• Limit access to Confidential Information to employees, contractors, or agents who have a need to know and who are bound by confidentiality obligations no less protective than this Agreement.",
        styles["NDABody"],
    ))
    story.append(Paragraph(
        "• Not reverse engineer, decompile, or disassemble any software or materials provided as part of the Project, except to the extent expressly permitted by applicable law.",
        styles["NDABody"],
    ))
    story.append(Paragraph(
        "• Return or destroy all Confidential Information upon Discloser's request or upon termination of this Agreement, and certify in writing upon request that such return or destruction has been completed.",
        styles["NDABody"],
    ))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("4. Term and Termination", styles["NDAHeading"]))
    story.append(Paragraph(
        "This Agreement is effective as of the Effective Date and continues for [NUMBER] years from the last disclosure of Confidential Information, "
        "or until terminated by either Party with [NUMBER] days written notice. Obligations with respect to Confidential Information survive termination for the period specified above or as required by law.",
        styles["NDABody"],
    ))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("5. No License; No Warranty", styles["NDAHeading"]))
    story.append(Paragraph(
        "No license or right under any patent, copyright, or other intellectual property is granted by this Agreement. "
        "Confidential Information is provided \"AS IS\"; Discloser makes no warranty regarding its accuracy or completeness.",
        styles["NDABody"],
    ))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("6. General", styles["NDAHeading"]))
    story.append(Paragraph(
        "<b>Governing Law:</b> This Agreement is governed by the laws of [JURISDICTION], without regard to conflict of laws principles.",
        styles["NDABody"],
    ))
    story.append(Paragraph(
        "<b>Entire Agreement:</b> This Agreement constitutes the entire agreement between the Parties regarding the subject matter and supersedes prior discussions and agreements.",
        styles["NDABody"],
    ))
    story.append(Paragraph("<b>Amendment:</b> Modifications must be in writing and signed by both Parties.", styles["NDABody"]))
    story.append(Paragraph("<b>Severability:</b> If any provision is held invalid, the remainder remains in effect.", styles["NDABody"]))
    story.append(Paragraph("<b>No Assignment:</b> Neither Party may assign this Agreement without the other's prior written consent.", styles["NDABody"]))
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("<b>DISCLOSING PARTY – Paafekt</b> (creating this NDA and requesting signature)", styles["NDAHeading"]))
    story.append(Paragraph("[Full legal entity name]", styles["NDABody"]))
    story.append(Spacer(1, 0.05 * inch))
    story.append(Paragraph("By: _________________________", styles["NDABody"]))
    story.append(Paragraph("Name: Jayamma", styles["NDABody"]))
    story.append(Paragraph("Title: [Title]", styles["NDABody"]))
    story.append(Paragraph("Date: _________________________", styles["NDABody"]))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("<b>RECEIVING PARTY</b> (sign below to accept)", styles["NDAHeading"]))
    story.append(Paragraph("[FULL LEGAL NAME / COMPANY NAME]", styles["NDABody"]))
    story.append(Spacer(1, 0.05 * inch))
    story.append(Paragraph("By: _________________________", styles["NDABody"]))
    story.append(Paragraph("Name: _________________________", styles["NDABody"]))
    story.append(Paragraph("Title: _________________________", styles["NDABody"]))
    story.append(Paragraph("Date: _________________________", styles["NDABody"]))

    doc.build(story, onFirstPage=add_letterhead_bands, onLaterPages=add_letterhead_bands)
    buffer.seek(0)
    return buffer


def main():
    if not LETTERHEAD_TOP.exists() or not LETTERHEAD_BOTTOM.exists():
        raise FileNotFoundError(
            f"Letterhead images not found. Expected:\n  {LETTERHEAD_TOP}\n  {LETTERHEAD_BOTTOM}"
        )

    nda_buffer = create_nda_abap_pages()

    output_path = Path(OUTPUT_PDF)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(nda_buffer.read())

    print(f"Created: {output_path}")
    print("Letterhead: top and bottom bands only (docs/letterhead/top.png, bottom.png)")
    print("NDA: ABAP AI Toolkit – Paafekt (Jayamma) as Disclosing Party requesting Recipient to sign.")


if __name__ == "__main__":
    main()

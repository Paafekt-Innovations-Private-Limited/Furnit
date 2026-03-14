#!/usr/bin/env python3
"""Produce a PDF with Profile content; Blackline end date = 6th March 2026, no 'work at Blackline now'."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

# Use built-in font to avoid missing fonts
styles = getSampleStyleSheet()
body = ParagraphStyle(
    name="Body",
    parent=styles["Normal"],
    fontSize=10,
    leading=12,
)
title_style = ParagraphStyle(
    name="Title",
    parent=styles["Heading1"],
    fontSize=16,
    spaceAfter=6,
)
heading_style = ParagraphStyle(
    name="Heading",
    parent=styles["Heading2"],
    fontSize=12,
    spaceAfter=4,
)

def text(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def main():
    out_path = "/Users/al/Downloads/Profile_updated.pdf"
    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    story = []

    story.append(Paragraph(text("Kishore Shivanna"), title_style))
    story.append(Paragraph(text("Bengaluru, Karnataka, India"), body))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("CONTACT", heading_style))
    story.append(Paragraph(text("kishore.shivanna@gmail.com | +91-7795002599 | www.linkedin.com/in/kishore-shivanna-0762429"), body))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("TOP SKILLS", heading_style))
    story.append(Paragraph(text("Software Development, Python, Swift, Kotlin, Java, Spring Boot, ML/AI, SHARP, YOLOe, RAG with LLM, Training LLM (e.g. ABAP), REST, Git, Android/iOS, Camera vision"), body))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("TOOLS USED", heading_style))
    story.append(Paragraph(text("Xcode, Android Studio, IntelliJ IDEA"), body))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Summary", heading_style))
    story.append(Paragraph(
        text("Over 20 years in IT with hands-on expertise in Java, Kotlin, Swift, Python, ML, and full-stack development. "
             "Until last year it was predominantly Java-based application development; work took a drastic turn as AI became central—Swift, Android, "
             "and camera vision were taken up in parallel to test and deploy AI capabilities on mobile and edge. "
             "Strong focus on AI/ML: SHARP, YOLOe, RAG with LLM, and training LLMs for programming languages (e.g. ABAP). "
             "Built web and mobile applications with Spring, Android, iOS, and cloud deployment (Amazon, GKE). "
             "Innovation-driven with experience in project management, technical design, and Agile delivery."),
        body,
    ))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Journey & Outlook", heading_style))
    story.append(Paragraph(
        text("Evolution of learning: from Java/J2EE to full-stack, mobile, then Python and AI/ML. "
             "I no longer see myself as someone with Java, J2EE skills anymore—AI changed that. "
             "The change is deliberate: I took up Python, on-device ML, and LLMs to solve real problems (vision, RAG, training for niche use cases like ABAP). "
             "Outlook: Companies will increasingly adopt AI/ML/LLM; engineers will need to train and tune models for niche domains and bigger challenges. "
             "Value lies in custom models, domain-specific training, and solving hard problems."),
        body,
    ))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Experience", heading_style))

    # Blackline: end date 6th March 2026; no "work in Blackline now"
    story.append(Paragraph(text("Blackline (IDC) Private Limited"), body))
    story.append(Paragraph(text("Senior Technical Manager"), body))
    story.append(Paragraph(text("February 2023 – 6th March 2026 · India"), body))
    story.append(Paragraph(
        text("Financial close & accounting automation. Core product development, requirement analysis, design and implementation. "
             "Java, Python, Ruby, Angular, RAG with LLM; BTP to GKE; SAP ERP, ABAP."),
        body,
    ))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph(text("Innovation & R&D — AI/ML & Mobile"), body))
    story.append(Paragraph(
        text("Pivot from Java to AI-driven development; Swift, Kotlin, Android, camera vision in parallel to test AI on mobile and edge. "
             "Python, ML, ONNX, SHARP, YOLOe; RAG with LLM; training LLM for programming (e.g. ABAP); on-device ML, Unreal Engine."),
        body,
    ))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph(text("Operative Media Private Limited — Senior Technical Lead"), body))
    story.append(Paragraph(text("June 2015 – February 2023 · 7+ years"), body))
    story.append(Paragraph(text("Media/ad-tech platform. Spring (REST, DWR), MySQL, Tomcat, SVN."), body))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph(text("Indecomm Global Services (Accelrys IDC) — Software Manager"), body))
    story.append(Paragraph(text("June 2010 – January 2012 & November 2013 – June 2015 · Bangalore"), body))
    story.append(Paragraph(text("Accelrys product development. Spring (REST), Grails, GWT, EJB 3, Oracle, JBoss, Tomcat, Perforce."), body))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph(text("Startup Ventures — Jeepnee, Bovo, Easy12buy — Project Lead"), body))
    story.append(Paragraph(text("December 2011 – November 2013"), body))
    story.append(Paragraph(text("E-commerce, same-day delivery. Neo4j, MySQL, Grails, Angular JS, Git."), body))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph(text("Infosys Technologies Limited — Technical Lead (October 2008 – June 2010)"), body))
    story.append(Paragraph(text("Finacle core banking. CTS, Fund Transfer, workflows."), body))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph(text("Manhattan Associates — Software Engineer (July 2007 – August 2008)"), body))
    story.append(Paragraph(text("Supply chain TMS. JSP, EJB, Hibernate, Oracle, WebLogic, WebSphere."), body))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph(text("GT Nexus — Software Engineer (June 2004 – June 2007)"), body))
    story.append(Paragraph(text("Global trade and logistics. JSP, EJB, Hibernate, MS-SQL, WebLogic."), body))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Education", heading_style))
    story.append(Paragraph(text("BE, Instrumentation Technology, 1999 – 2003 · 65%"), body))
    story.append(Paragraph(text("Sri Jayachamarajendra College of Engineering, Mysore (V.T.U.)"), body))

    doc.build(story)
    print(out_path)

if __name__ == "__main__":
    main()

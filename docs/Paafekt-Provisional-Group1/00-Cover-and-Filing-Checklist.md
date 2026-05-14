# PAAFEKT INC.

## USPTO PROVISIONAL PATENT APPLICATION BUNDLE

### Group 1 — Paafekt Mobile Application (6 Patents)

---

## COVER PAGE & FILING CHECKLIST

**Applicant:** Paafekt Inc.  
**State of Incorporation:** Delaware C-Corporation  
**Entity ID:** 10593049  
**Inventor:** Kishore Shivanna (sole inventor)  
**Filing Type:** Provisional Patent Application  
**Entity Status:** Micro Entity ($320 per provisional filing)  
**Filing Date:** [DATE OF FILING]  
**Prepared By:** Self-Filed / Pro Se  

---

## USPTO FILING CHECKLIST (Per Provisional Application)

For each of the 6 provisional applications below, the following items must be submitted to the USPTO via EFS-Web / Patent Center:

- [ ] **Application Data Sheet (ADS)** — Form PTO/SB/14 — list inventor, applicant, title, correspondence address
- [ ] **Provisional Patent Application Specification** — This document (one per patent)
- [ ] **Filing Fee** — $320 per provisional (Micro Entity rate, as of 2026)
- [ ] **Micro Entity Certification** — Form PTO/SB/15A (Gross Income Basis) — certify that inventor's gross income does not exceed the micro entity threshold
- [ ] **Cover Sheet** — Include title, inventor name(s), correspondence address
- [ ] **Drawings/Figures** — Optional for provisional; can be added. Reference figures are described in text and can be attached as informal sketches

### Important Notes

- Provisional applications are NOT examined. They establish a priority date only.
- The applicant has 12 months from the provisional filing date to file a corresponding non-provisional (utility) patent application claiming priority to this provisional.
- Provisional applications are never published and become abandoned after 12 months unless a non-provisional is filed.
- Each patent below should be filed as a SEPARATE provisional application, each with its own filing fee and ADS.

---

## Generated Word documents (`.docx`)

In this folder, pandoc can build (or has built):

- **`Paafekt-Provisional-Group1-MASTER-All-In-One.docx`** — cover, all six specifications, and appendices in one file (convenient for review and global find/replace).
- **`Paafekt-Provisional-Group1-SPEC-01-ONLY.docx` … `SPEC-06-ONLY.docx`** — one document per provisional (use each as the specification PDF for that USPTO filing).

Rebuild from the Markdown sources (example for the master file):

```bash
cd docs/Paafekt-Provisional-Group1
pandoc 00-Cover-and-Filing-Checklist.md \
  01-Provisional-On-Device-3D-Room-Reconstruction.md \
  02-Provisional-Neural-Segmentation-Architectural-Surfaces.md \
  03-Provisional-Centimetre-Accurate-Spatial-Measurement.md \
  04-Provisional-AR-Furniture-Placement-Visualization.md \
  05-Provisional-On-Device-Neural-Processing-Architecture.md \
  06-Provisional-Integrated-Mobile-Application-System.md \
  07-Appendices-Glossary-Figures-Filing-Instructions.md \
  -o Paafekt-Provisional-Group1-MASTER-All-In-One.docx --standalone
```

Per-specification DOCX: `pandoc 01-Provisional-On-Device-3D-Room-Reconstruction.md -o Paafekt-Provisional-Group1-SPEC-01-ONLY.docx --standalone` (repeat for `02`–`06`).

## How to use these files in Microsoft Word

1. **Recommended:** Open the `.docx` files above, or open each numbered specification (`01`–`06`) as `.md` in Word (File → Open → choose the `.md` file, or paste the contents into a blank document).
2. In Word, apply styles: **Heading 1** for top-level patent headers, **Heading 2** for major sections (Field, Background, Summary, Detailed Description, Claims, Abstract), **Heading 3** for subsections under Detailed Description, **Normal** for body text.
3. Before each **Claims** section, consider **multilevel list** numbering for claim dependencies.
4. For **Appendix B** table, use Word’s **Insert → Table** or paste from `07-Appendices.md` and apply a table style.
5. **USPTO filing:** You will typically upload **one PDF per provisional**; export each specification (and that filing’s ADS and micro-entity form) to PDF separately.

---

*End of cover and checklist — specifications follow in separate files.*

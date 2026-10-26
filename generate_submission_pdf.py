"""
generate_submission_pdf.py
==========================
Compiles the final mandatory single PDF submission file required for BITS Pilani
M.Tech (AIML) Machine Learning Assignment 2.

Mandatory Order of Sections (as specified in assignment guidelines):
1. GitHub Repository Link (containing source code, requirements.txt, README.md, test_data.csv)
2. Live Streamlit App Link (Streamlit Community Cloud clickable link)
3. BITS Virtual Lab Screenshot (embedded image if present, or clear placeholder)
4. Full GitHub README Content (Problem statement, Dataset description, Models used, Comparison table, Observations, Winner)
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def build_pdf(
    output_filename="ML_Assignment_2_Submission.pdf",
    student_name="Naveen Joy",
    student_id="[Insert BITS Student ID]",
    github_url="https://github.com/your-username/bits-ml-assignment-2",
    streamlit_url="https://bits-ml-assignment-2.streamlit.app",
    screenshot_path="bits_lab_screenshot.png"
):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1e3a8a'),
        alignment=1, # Center
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#475569'),
        alignment=1,
        spaceAfter=15
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1e3a8a'),
        spaceBefore=12,
        spaceAfter=6
    )

    subsection_heading = ParagraphStyle(
        'SubSectionHeading',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=6
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1e40af')
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#0f172a'),
        alignment=1
    )

    table_cell_left = ParagraphStyle(
        'TableCellLeft',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#0f172a')
    )

    story = []

    # Title & Header
    story.append(Paragraph("BITS Pilani • Work Integrated Learning Programmes (WILP)", title_style))
    story.append(Paragraph("M.Tech (AIML) — Machine Learning (Assignment - 2)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1e3a8a'), spaceAfter=12))

    # Student Info Table
    info_data = [
        [
            Paragraph(f"<b>Candidate Name:</b> {student_name}", body_style),
            Paragraph(f"<b>Student ID:</b> {student_id}", body_style)
        ],
        [
            Paragraph("<b>Programme:</b> M.Tech in AI & ML", body_style),
            Paragraph("<b>Topic:</b> Multi-Model Classification & Streamlit Cloud", body_style)
        ]
    ]
    info_table = Table(info_data, colWidths=[260, 260])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 14))

    # =========================================================================
    # SECTION 1: Mandatory Submission Links
    # =========================================================================
    story.append(Paragraph("1. Mandatory Submission Links", section_heading))
    
    links_data = [
        [
            Paragraph("<b>Deliverable</b>", table_header_style),
            Paragraph("<b>Clickable URL / Submission Link</b>", table_header_style)
        ],
        [
            Paragraph("<b>GitHub Repository Link</b><br/><i>(Source code, requirements.txt, README.md, test_data.csv)</i>", table_cell_left),
            Paragraph(f'<link href="{github_url}"><font color="#1d4ed8"><u>{github_url}</u></font></link>', table_cell_left)
        ],
        [
            Paragraph("<b>Live Streamlit App Link</b><br/><i>(Deployed on Streamlit Community Cloud)</i>", table_cell_left),
            Paragraph(f'<link href="{streamlit_url}"><font color="#1d4ed8"><u>{streamlit_url}</u></font></link>', table_cell_left)
        ]
    ]
    links_table = Table(links_data, colWidths=[200, 320])
    links_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.8, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
    ]))
    story.append(links_table)
    story.append(Spacer(1, 14))

    # =========================================================================
    # SECTION 2: Screenshot of Execution on BITS Virtual Lab
    # =========================================================================
    story.append(Paragraph("2. Proof of Execution on BITS Virtual Lab [1 Mark]", section_heading))
    story.append(Paragraph("As per assignment instructions, execution on the BITS Virtual Lab environment is verified below:", body_style))

    if os.path.exists(screenshot_path):
        img = Image(screenshot_path, width=6.5*inch, height=3.6*inch)
        story.append(img)
    else:
        # Informative placeholder box
        placeholder_content = [
            [Paragraph("<br/><br/><b>[ BITS VIRTUAL LAB SCREENSHOT PLACEHOLDER ]</b><br/><br/>"
                       "<i>Take a screenshot of your assignment execution inside BITS Virtual Lab and save it as:<br/>"
                       "<b>bits_lab_screenshot.png</b> in this project folder, then re-run:</i><br/>"
                       "<code>python generate_submission_pdf.py</code><br/>"
                       "<i>It will automatically embed your screenshot here.</i><br/><br/>",
                       ParagraphStyle('Pbox', parent=styles['Normal'], alignment=1, textColor=colors.HexColor('#64748b'), fontSize=10, leading=15))]
        ]
        ph_table = Table(placeholder_content, colWidths=[520], rowHeights=[140])
        ph_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f1f5f9')),
            ('BOX', (0, 0), (-1, -1), 1.5, colors.HexColor('#94a3b8')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(ph_table)

    story.append(Spacer(1, 14))

    # =========================================================================
    # SECTION 3: GitHub README Content (Section 3 - Step 5)
    # =========================================================================
    story.append(Paragraph("3. GitHub README Content (Assignment Details & Results)", section_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e1'), spaceAfter=10))

    # 3.a Problem Statement
    story.append(Paragraph("a. Problem Statement", subsection_heading))
    p_stmt = (
        "In modern digital commerce, understanding customer browsing behavior and accurately predicting whether an "
        "online session will culminate in a purchase transaction is critical for revenue growth, dynamic personalized marketing, "
        "and conversion rate optimization (CRO). The goal of this project is to develop, evaluate, and deploy an end-to-end "
        "Machine Learning classification system that predicts Online Shoppers Purchasing Intention (Revenue: True/False). "
        "Utilizing a multi-model comparative framework, we implement, benchmark, and deploy five distinct classification algorithms: "
        "Logistic Regression, Decision Tree Classifier, K-Nearest Neighbor (KNN) Classifier, Naive Bayes Classifier (Gaussian NB), "
        "and Support Vector Machine (SVM with RBF Kernel). The models are evaluated across six key metrics: Accuracy, AUC-ROC, "
        "Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC)."
    )
    story.append(Paragraph(p_stmt, body_style))
    story.append(Spacer(1, 6))

    # 3.b Dataset Description
    story.append(Paragraph("b. Dataset Description [1 Mark]", subsection_heading))
    p_data = (
        "<b>Repository:</b> UCI Machine Learning Repository / Kaggle<br/>"
        "<b>Dataset:</b> Online Shoppers Purchasing Intention Dataset<br/>"
        "<b>Total Instances:</b> 12,330 sessions (satisfies requirement: minimum 500)<br/>"
        "<b>Total Features:</b> 17 input attributes + 1 target attribute = 18 total (satisfies requirement: minimum 12 features)<br/>"
        "<b>Classification Type:</b> Binary Classification<br/>"
        "<b>Target Variable:</b> <code>Revenue</code> (True: 15.47%, False: 84.53%)<br/>"
        "<b>Key Attributes:</b> Administrative, Administrative_Duration, Informational, Informational_Duration, "
        "ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, SpecialDay, Month, "
        "OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend."
    )
    story.append(Paragraph(p_data, body_style))
    story.append(Spacer(1, 6))

    # 3.c Models Used & Comparison Table
    story.append(Paragraph("c. Models Used & Evaluation Comparison Table [5 Marks]", subsection_heading))
    story.append(Paragraph("All 5 models were trained on the identical 80% training split (9,864 samples) and tested on the 20% holdout split (2,466 samples) with standardized ColumnTransformer preprocessing:", body_style))

    comp_table_data = [
        [
            Paragraph("<b>ML Model Name</b>", table_header_style),
            Paragraph("<b>Accuracy</b>", table_header_style),
            Paragraph("<b>AUC</b>", table_header_style),
            Paragraph("<b>Precision</b>", table_header_style),
            Paragraph("<b>Recall</b>", table_header_style),
            Paragraph("<b>F1</b>", table_header_style),
            Paragraph("<b>MCC</b>", table_header_style)
        ],
        [
            Paragraph("<b>Logistic Regression</b>", table_cell_left),
            Paragraph("0.8808", table_cell_style),
            Paragraph("0.8828", table_cell_style),
            Paragraph("0.7366", table_cell_style),
            Paragraph("0.3586", table_cell_style),
            Paragraph("0.4824", table_cell_style),
            Paragraph("0.4592", table_cell_style)
        ],
        [
            Paragraph("<b>Decision Tree</b> <i>[Winner]</i>", table_cell_left),
            Paragraph("<b>0.8994</b>", table_cell_style),
            Paragraph("<b>0.9178</b>", table_cell_style),
            Paragraph("0.7107", table_cell_style),
            Paragraph("0.5916", table_cell_style),
            Paragraph("<b>0.6457</b>", table_cell_style),
            Paragraph("<b>0.5910</b>", table_cell_style)
        ],
        [
            Paragraph("<b>KNN</b>", table_cell_left),
            Paragraph("0.8751", table_cell_style),
            Paragraph("0.8101", table_cell_style),
            Paragraph("0.6907", table_cell_style),
            Paragraph("0.3508", table_cell_style),
            Paragraph("0.4653", table_cell_style),
            Paragraph("0.4327", table_cell_style)
        ],
        [
            Paragraph("<b>Naive Bayes</b>", table_cell_left),
            Paragraph("0.7810", table_cell_style),
            Paragraph("0.8065", table_cell_style),
            Paragraph("0.3799", table_cell_style),
            Paragraph("<b>0.6545</b>", table_cell_style),
            Paragraph("0.4808", table_cell_style),
            Paragraph("0.3752", table_cell_style)
        ],
        [
            Paragraph("<b>Support Vector Machine</b>", table_cell_left),
            Paragraph("0.8873", table_cell_style),
            Paragraph("0.8824", table_cell_style),
            Paragraph("0.7047", table_cell_style),
            Paragraph("0.4686", table_cell_style),
            Paragraph("0.5629", table_cell_style),
            Paragraph("0.5149", table_cell_style)
        ]
    ]

    comp_table = Table(comp_table_data, colWidths=[140, 60, 60, 65, 60, 60, 65])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#fef9c3')), # highlight Decision Tree
    ]))
    story.append(comp_table)
    story.append(Spacer(1, 10))

    # 3.d Observations Table
    story.append(Paragraph("d. Observations on Model Performance [3 Marks]", subsection_heading))

    obs_table_data = [
        [
            Paragraph("<b>ML Model Name</b>", table_header_style),
            Paragraph("<b>Observation about model performance</b>", table_header_style)
        ],
        [
            Paragraph("<b>Logistic Regression</b>", table_cell_left),
            Paragraph("Solid linear baseline with high accuracy (88.08%) and high precision (73.66%), but low recall (35.86%) due to inability to model non-linear session threshold interactions.", table_cell_left)
        ],
        [
            Paragraph("<b>Decision Tree</b>", table_cell_left),
            Paragraph("Best overall classifier. Captures hierarchical visitor engagement rules (e.g. PageValues thresholds) achieving top AUC (0.9178), top F1 (0.6457), top MCC (0.5910), and top Accuracy (89.94%).", table_cell_left)
        ],
        [
            Paragraph("<b>KNN</b>", table_cell_left),
            Paragraph("Decent precision (69.07%) and 87.51% accuracy, but suffers from metric sparsity in high-dimensional one-hot encoded space resulting in low recall (35.08%) and lower AUC (0.8101).", table_cell_left)
        ],
        [
            Paragraph("<b>Naive Bayes</b>", table_cell_left),
            Paragraph("Demonstrates high recall (65.45%), effectively discovering converters, but suffers from low precision (37.99%) and lower accuracy (78.10%) because the feature independence assumption fails on correlated session variables.", table_cell_left)
        ],
        [
            Paragraph("<b>Support Vector Machine</b>", table_cell_left),
            Paragraph("Strong runner-up with 88.73% accuracy and 0.8824 AUC. The RBF kernel captures non-linear boundary separation well, maintaining 70.47% precision and 0.5149 MCC.", table_cell_left)
        ]
    ]

    obs_table = Table(obs_table_data, colWidths=[140, 380])
    obs_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
    ]))
    story.append(obs_table)
    story.append(Spacer(1, 10))

    # Winner Analysis
    story.append(Paragraph("<b>Overall Winner for the Dataset: Decision Tree Classifier</b>", subsection_heading))
    winner_text = (
        "The Decision Tree classifier is the indisputable winner across AUC (0.9178), F1-Score (0.6457), "
        "Matthews Correlation Coefficient (0.5910), and Accuracy (89.94%). Web browsing and purchase intent behavior "
        "inherently follow sequential, condition-based decision steps (e.g., <i>if PageValues > threshold AND BounceRates < limit</i>), "
        "which tree-based architectures naturally represent with high interpretability and minimal distortion."
    )
    story.append(Paragraph(winner_text, body_style))

    doc.build(story)
    print(f"-> Successfully generated submission PDF: '{output_filename}'")

if __name__ == '__main__':
    build_pdf()

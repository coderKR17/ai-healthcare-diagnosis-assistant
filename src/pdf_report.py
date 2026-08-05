"""PDF report generation module for the AI-Powered Healthcare Diagnosis
Assistant.

This module provides a reusable, framework-independent function to
generate a professional, well-formatted medical report in PDF format
using ``reportlab``. The report consolidates patient information, BMI
assessment, predicted disease, medicine recommendations, lab test
recommendations, and diagnosis history into a single document.

Typical usage example:
    from src.pdf_report import generate_medical_report

    output_path = generate_medical_report(
        patient=patient,
        bmi_data={
            "bmi": 22.86,
            "category": "Normal",
            "health_risk": "Low Risk",
            "health_tip": "Maintain your current weight...",
        },
        predicted_disease="Migraine",
        medicine_data={
            "disease": "Migraine",
            "common_medicines": ["Analgesics", "Triptans"],
            "precautions": ["Avoid known triggers"],
        },
        lab_test_data={
            "disease": "Migraine",
            "recommended_tests": ["MRI Brain"],
            "purpose": ["Rule out structural brain abnormalities"],
        },
        history=[],
        output_path="reports/patient_report.pdf",
    )
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DISCLAIMER_TEXT: Final[str] = (
    "This report is generated for educational and informational purposes "
    "only. It does not constitute a medical diagnosis, prescription, or "
    "treatment plan. Please consult a qualified healthcare professional "
    "for accurate diagnosis and appropriate treatment."
)


class PDFReportError(Exception):
    """Raised when PDF report generation fails.

    This exception wraps any underlying error encountered while building
    or writing the medical report PDF document.
    """


def _build_styles() -> dict[str, ParagraphStyle]:
    """Build and return the paragraph styles used throughout the report.

    Returns:
        A dictionary mapping style names to ``ParagraphStyle`` instances.
    """
    base_styles = getSampleStyleSheet()

    styles: dict[str, ParagraphStyle] = {
        "title": ParagraphStyle(
            name="ReportTitle",
            parent=base_styles["Title"],
            fontSize=18,
            spaceAfter=12,
            textColor=colors.HexColor("#1F4E79"),
        ),
        "section_heading": ParagraphStyle(
            name="SectionHeading",
            parent=base_styles["Heading2"],
            fontSize=13,
            spaceBefore=14,
            spaceAfter=6,
            textColor=colors.HexColor("#1F4E79"),
        ),
        "body": ParagraphStyle(
            name="BodyText",
            parent=base_styles["Normal"],
            fontSize=10,
            leading=14,
        ),
        "disclaimer": ParagraphStyle(
            name="Disclaimer",
            parent=base_styles["Normal"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#666666"),
        ),
    }
    return styles


def _build_table(
    data: list[list[str]],
    styles: dict[str, ParagraphStyle],
    col_widths: list[float] | None = None,
) -> Table:
    """Build a styled table for use within the report.

    Args:
        data: A list of rows, where each row is a list of cell strings.
        styles: The dictionary of paragraph styles used for report text.
        col_widths: Optional list of column widths in points.

    Returns:
        A configured ``Table`` instance.
    """
    table = Table(data, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8FB")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _extract_patient_fields(patient: Any) -> dict[str, str]:
    """Extract required display fields from a patient object.

    Args:
        patient: A patient object exposing ``full_name``, ``age``,
            ``gender``, ``height_cm``, ``weight_kg``, and
            ``blood_group`` attributes.

    Returns:
        A dictionary of patient display fields as strings.

    Raises:
        PDFReportError: If the patient object is missing required
            attributes.
    """
    try:
        return {
            "Name": str(patient.full_name),
            "Age": str(patient.age),
            "Gender": str(patient.gender),
            "Height (cm)": str(patient.height_cm),
            "Weight (kg)": str(patient.weight_kg),
            "Blood Group": str(patient.blood_group),
        }
    except AttributeError as exc:
        message = "Patient object is missing one or more required fields."
        logger.error(message)
        raise PDFReportError(message) from exc


def generate_medical_report(
    patient: Any,
    bmi_data: dict[str, Any],
    predicted_disease: str,
    medicine_data: dict[str, Any],
    lab_test_data: dict[str, Any],
    history: list[dict[str, Any]],
    output_path: str,
) -> str:
    """Generate a professional medical report PDF.

    Args:
        patient: A patient object exposing ``full_name``, ``age``,
            ``gender``, ``height_cm``, ``weight_kg``, and
            ``blood_group`` attributes.
        bmi_data: A dictionary containing 'bmi', 'category',
            'health_risk', and 'health_tip' keys.
        predicted_disease: The name of the predicted disease.
        medicine_data: A dictionary containing 'disease',
            'common_medicines', and 'precautions' keys.
        lab_test_data: A dictionary containing 'disease',
            'recommended_tests', and 'purpose' keys.
        history: A list of history record dictionaries, each containing
            'patient_name', 'disease', 'symptoms', 'date', and 'time'.
        output_path: The file path where the generated PDF should be
            saved.

    Returns:
        The path to the generated PDF report file.

    Raises:
        PDFReportError: If the report cannot be generated or saved.
    """
    logger.info("Starting medical report generation.")

    try:
        styles = _build_styles()
        patient_fields = _extract_patient_fields(patient)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        document = SimpleDocTemplate(
            str(output_file),
            pagesize=A4,
            topMargin=1.5 * cm,
            bottomMargin=1.5 * cm,
            leftMargin=1.5 * cm,
            rightMargin=1.5 * cm,
        )

        elements: list[Any] = []

        # 1. Project Title
        elements.append(
            Paragraph("AI-Powered Healthcare Diagnosis Assistant", styles["title"])
        )
        elements.append(Paragraph("Medical Diagnosis Report", styles["body"]))
        elements.append(Spacer(1, 12))

        # 2. Patient Information
        elements.append(Paragraph("Patient Information", styles["section_heading"]))
        patient_table_data = [["Field", "Value"]] + [
            [field, value] for field, value in patient_fields.items()
        ]
        elements.append(
            _build_table(patient_table_data, styles, col_widths=[150, 300])
        )

        # 3. BMI Information
        elements.append(Paragraph("BMI & Health Risk Assessment", styles["section_heading"]))
        bmi_table_data = [
            ["Field", "Value"],
            ["BMI", str(bmi_data.get("bmi", "-"))],
            ["Category", str(bmi_data.get("category", "-"))],
            ["Health Risk", str(bmi_data.get("health_risk", "-"))],
            ["Health Tip", str(bmi_data.get("health_tip", "-"))],
        ]
        elements.append(_build_table(bmi_table_data, styles, col_widths=[150, 300]))

        # 4. Predicted Disease
        elements.append(Paragraph("Predicted Disease", styles["section_heading"]))
        elements.append(Paragraph(predicted_disease or "Not Available", styles["body"]))

        # 5. Medicine Recommendation
        elements.append(Paragraph("Medicine Recommendation", styles["section_heading"]))
        medicines = medicine_data.get("common_medicines", [])
        precautions = medicine_data.get("precautions", [])
        elements.append(Paragraph("<b>Common Medicines:</b>", styles["body"]))
        elements.append(
            Paragraph(
                "<br/>".join(f"- {item}" for item in medicines) or "Not Available",
                styles["body"],
            )
        )
        elements.append(Paragraph("<b>Precautions:</b>", styles["body"]))
        elements.append(
            Paragraph(
                "<br/>".join(f"- {item}" for item in precautions) or "Not Available",
                styles["body"],
            )
        )

        # 6. Recommended Lab Tests
        elements.append(Paragraph("Recommended Lab Tests", styles["section_heading"]))
        tests = lab_test_data.get("recommended_tests", [])
        purposes = lab_test_data.get("purpose", [])
        elements.append(Paragraph("<b>Tests:</b>", styles["body"]))
        elements.append(
            Paragraph(
                "<br/>".join(f"- {item}" for item in tests) or "Not Available",
                styles["body"],
            )
        )
        elements.append(Paragraph("<b>Purpose:</b>", styles["body"]))
        elements.append(
            Paragraph(
                "<br/>".join(f"- {item}" for item in purposes) or "Not Available",
                styles["body"],
            )
        )

        # 7. Disease History
        elements.append(Paragraph("Disease History", styles["section_heading"]))
        if history:
            history_table_data = [["Patient", "Disease", "Symptoms", "Date", "Time"]]
            for record in history:
                history_table_data.append(
                    [
                        str(record.get("patient_name", "-")),
                        str(record.get("disease", "-")),
                        ", ".join(record.get("symptoms", [])),
                        str(record.get("date", "-")),
                        str(record.get("time", "-")),
                    ]
                )
            elements.append(
                _build_table(
                    history_table_data,
                    styles,
                    col_widths=[80, 80, 150, 70, 60],
                )
            )
        else:
            elements.append(Paragraph("No previous history records available.", styles["body"]))

        # 8. Medical Disclaimer
        elements.append(Spacer(1, 16))
        elements.append(Paragraph("Medical Disclaimer", styles["section_heading"]))
        elements.append(Paragraph(DISCLAIMER_TEXT, styles["disclaimer"]))

        # 9. Report Generation Date & Time
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elements.append(Spacer(1, 10))
        elements.append(
            Paragraph(f"Report Generated On: {generated_at}", styles["disclaimer"])
        )

        document.build(elements)

    except PDFReportError:
        raise
    except Exception as exc:
        message = "Failed to generate medical report PDF."
        logger.error(message)
        raise PDFReportError(message) from exc

    logger.info("Medical report generated successfully at: %s", output_path)
    return str(output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    class _SamplePatient:
        """Minimal sample patient object for local testing."""

        full_name = "John Doe"
        age = 30
        gender = "Male"
        height_cm = 175.0
        weight_kg = 70.0
        blood_group = "O+"

    try:
        result_path = generate_medical_report(
            patient=_SamplePatient(),
            bmi_data={
                "bmi": 22.86,
                "category": "Normal",
                "health_risk": "Low Risk",
                "health_tip": "Maintain your current weight through a balanced diet.",
            },
            predicted_disease="Migraine",
            medicine_data={
                "disease": "Migraine",
                "common_medicines": ["Analgesics", "Triptans"],
                "precautions": ["Avoid known triggers such as bright lights"],
            },
            lab_test_data={
                "disease": "Migraine",
                "recommended_tests": ["MRI Brain"],
                "purpose": ["Rule out structural brain abnormalities"],
            },
            history=[
                {
                    "patient_name": "John Doe",
                    "disease": "Common Cold",
                    "symptoms": ["cough", "sneezing"],
                    "date": "2026-08-01",
                    "time": "10:15:00",
                }
            ],
            output_path="reports/sample_patient_report.pdf",
        )
        logger.info("Report generated at: %s", result_path)
    except PDFReportError:
        logger.exception("PDF report generation failed.")
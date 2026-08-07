"""Email medical report module for the AI-Powered Healthcare Diagnosis
Assistant.

This module provides a reusable, framework-independent function to send
a generated medical report PDF to a patient's email address via SMTP.

Typical usage example:
    from src.email_report import send_medical_report

    success = send_medical_report(
        recipient_email="patient@example.com",
        pdf_path="reports/medical_report.pdf",
        sender_email="clinic@example.com",
        sender_password="app_specific_password",
    )
"""

from __future__ import annotations

import logging
import re
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

EMAIL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
)

SMTP_SERVER: Final[str] = "smtp.gmail.com"
SMTP_PORT: Final[int] = 587

EMAIL_SUBJECT: Final[str] = "AI Healthcare Diagnosis Assistant Medical Report"
EMAIL_BODY: Final[str] = (
    "Dear Patient,\n\n"
    "Please find attached your generated medical report.\n"
    "This report is intended only for educational purposes.\n\n"
    "Regards,\n"
    "AI Healthcare Diagnosis Assistant"
)


class EmailReportError(Exception):
    """Raised when sending the medical report email fails.

    This exception is raised whenever the recipient email is invalid,
    the PDF report file does not exist, or the email cannot be sent due
    to an SMTP or connection error.
    """


def _validate_email(email: str) -> str:
    """Validate an email address format.

    Args:
        email: The email address to validate.

    Returns:
        The stripped, validated email address.

    Raises:
        EmailReportError: If the email is empty or improperly formatted.
    """
    if not isinstance(email, str) or not email.strip():
        message = "Recipient email cannot be empty."
        logger.error(message)
        raise EmailReportError(message)

    normalized_email = email.strip()

    if not EMAIL_PATTERN.match(normalized_email):
        message = f"Invalid email address format: '{normalized_email}'."
        logger.error(message)
        raise EmailReportError(message)

    logger.info("Recipient email validated: %s", normalized_email)
    return normalized_email


def _validate_pdf_path(pdf_path: str) -> Path:
    """Validate that the PDF report file exists.

    Args:
        pdf_path: The path to the PDF report file.

    Returns:
        The validated ``Path`` instance pointing to the PDF file.

    Raises:
        EmailReportError: If the PDF path is empty, does not exist, or
            is not a file.
    """
    if not pdf_path or not str(pdf_path).strip():
        message = "PDF report path cannot be empty."
        logger.error(message)
        raise EmailReportError(message)

    resolved_path = Path(pdf_path)

    if not resolved_path.exists():
        message = f"PDF report file not found at: {resolved_path}"
        logger.error(message)
        raise EmailReportError(message)

    if not resolved_path.is_file():
        message = f"PDF report path is not a file: {resolved_path}"
        logger.error(message)
        raise EmailReportError(message)

    logger.info("PDF report file found at: %s", resolved_path)
    return resolved_path


def _build_email_message(
    recipient_email: str,
    sender_email: str,
    pdf_path: Path,
) -> EmailMessage:
    """Build the email message with the PDF report attached.

    Args:
        recipient_email: The validated recipient email address.
        sender_email: The sender's email address.
        pdf_path: The validated path to the PDF report file.

    Returns:
        A fully constructed ``EmailMessage`` ready to be sent.

    Raises:
        EmailReportError: If the PDF file cannot be read or attached.
    """
    message = EmailMessage()
    message["Subject"] = EMAIL_SUBJECT
    message["From"] = sender_email
    message["To"] = recipient_email
    message.set_content(EMAIL_BODY)

    try:
        pdf_bytes = pdf_path.read_bytes()
    except OSError as exc:
        error_message = f"Failed to read PDF report file: {pdf_path}"
        logger.error(error_message)
        raise EmailReportError(error_message) from exc

    message.add_attachment(
        pdf_bytes,
        maintype="application",
        subtype="pdf",
        filename=pdf_path.name,
    )

    logger.info("Email message constructed with attachment: %s", pdf_path.name)
    return message


def send_medical_report(
    recipient_email: str,
    pdf_path: str,
    sender_email: str,
    sender_password: str,
) -> bool:
    """Send a medical report PDF to a patient via email.

    Args:
        recipient_email: The patient's email address.
        pdf_path: The path to the generated medical report PDF file.
        sender_email: The sender's email address (used for SMTP
            authentication and the 'From' header).
        sender_password: The sender's email account password or
            app-specific password.

    Returns:
        True if the email was sent successfully.

    Raises:
        EmailReportError: If the recipient email is invalid, the PDF
            file does not exist, or the email fails to send due to an
            SMTP or connection error.
    """
    logger.info("Starting medical report email dispatch.")

    validated_recipient_email = _validate_email(recipient_email)
    validated_pdf_path = _validate_pdf_path(pdf_path)

    if not sender_email or not sender_email.strip():
        message = "Sender email cannot be empty."
        logger.error(message)
        raise EmailReportError(message)

    if not sender_password:
        message = "Sender password cannot be empty."
        logger.error(message)
        raise EmailReportError(message)

    email_message = _build_email_message(
        recipient_email=validated_recipient_email,
        sender_email=sender_email.strip(),
        pdf_path=validated_pdf_path,
    )

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp_connection:
            smtp_connection.starttls()
            smtp_connection.login(sender_email.strip(), sender_password)
            smtp_connection.send_message(email_message)
    except smtplib.SMTPAuthenticationError as exc:
        message = "SMTP authentication failed. Check sender credentials."
        logger.error(message)
        raise EmailReportError(message) from exc
    except smtplib.SMTPException as exc:
        message = "Failed to send medical report email due to an SMTP error."
        logger.error(message)
        raise EmailReportError(message) from exc
    except OSError as exc:
        message = "Failed to connect to the SMTP server."
        logger.error(message)
        raise EmailReportError(message) from exc

    logger.info(
        "Medical report email sent successfully to: %s", validated_recipient_email
    )
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        result = send_medical_report(
            recipient_email="patient@example.com",
            pdf_path="reports/medical_report.pdf",
            sender_email="clinic@example.com",
            sender_password="app_specific_password",
        )
        logger.info("Email sent successfully: %s", result)
    except EmailReportError:
        logger.exception("Failed to send medical report email.")
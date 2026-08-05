"""Dashboard summary module for the AI-Powered Healthcare Diagnosis
Assistant.

This module provides a reusable, framework-independent function to
consolidate patient, BMI, predicted disease, and history data into a
single summary dictionary suitable for display on a dashboard.

Typical usage example:
    from src.dashboard import generate_dashboard_summary

    summary = generate_dashboard_summary(
        patient=patient,
        bmi_data={"bmi": 22.86, "category": "Normal"},
        predicted_disease="Migraine",
        history=[],
    )
"""

from __future__ import annotations

import logging
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DashboardError(Exception):
    """Raised when dashboard summary generation fails.

    This exception is raised whenever the provided patient, BMI,
    predicted disease, or history data is invalid or incomplete.
    """


def _validate_patient(patient: Any) -> None:
    """Validate that the patient object contains required attributes.

    Args:
        patient: The patient object to validate.

    Raises:
        DashboardError: If the patient object is ``None`` or is missing
            required attributes.
    """
    if patient is None:
        message = "Patient data is required to generate a dashboard summary."
        logger.error(message)
        raise DashboardError(message)

    required_attributes = ("full_name", "age", "gender")
    missing_attributes = [
        attribute
        for attribute in required_attributes
        if not hasattr(patient, attribute)
    ]

    if missing_attributes:
        message = (
            f"Patient object is missing required attributes: "
            f"{missing_attributes}."
        )
        logger.error(message)
        raise DashboardError(message)


def _validate_bmi_data(bmi_data: dict[str, Any]) -> None:
    """Validate that the BMI data dictionary contains required keys.

    Args:
        bmi_data: The BMI data dictionary to validate.

    Raises:
        DashboardError: If ``bmi_data`` is not a dictionary or is missing
            required keys.
    """
    if not isinstance(bmi_data, dict):
        message = "BMI data must be provided as a dictionary."
        logger.error(message)
        raise DashboardError(message)

    required_keys = {"bmi", "category"}
    missing_keys = required_keys - bmi_data.keys()

    if missing_keys:
        message = f"BMI data is missing required keys: {sorted(missing_keys)}."
        logger.error(message)
        raise DashboardError(message)


def _validate_predicted_disease(predicted_disease: str) -> None:
    """Validate that the predicted disease value is a non-empty string.

    Args:
        predicted_disease: The predicted disease name to validate.

    Raises:
        DashboardError: If the predicted disease is empty or not a
            string.
    """
    if not isinstance(predicted_disease, str) or not predicted_disease.strip():
        message = "Predicted disease must be a non-empty string."
        logger.error(message)
        raise DashboardError(message)


def _validate_history(history: list[Any]) -> None:
    """Validate that the history value is a list.

    Args:
        history: The history data to validate.

    Raises:
        DashboardError: If ``history`` is not a list.
    """
    if not isinstance(history, list):
        message = "History must be provided as a list."
        logger.error(message)
        raise DashboardError(message)


def generate_dashboard_summary(
    patient: Any,
    bmi_data: dict[str, Any],
    predicted_disease: str,
    history: list[Any],
) -> dict[str, Any]:
    """Generate a consolidated dashboard summary.

    Args:
        patient: A patient object exposing ``full_name``, ``age``, and
            ``gender`` attributes.
        bmi_data: A dictionary containing at least 'bmi' and 'category'
            keys.
        predicted_disease: The name of the predicted disease.
        history: A list of diagnosis history records.

    Returns:
        A dictionary with the following keys:
            - "patient_name": The patient's full name.
            - "age": The patient's age.
            - "gender": The patient's gender.
            - "bmi": The patient's BMI value.
            - "bmi_category": The patient's BMI category.
            - "predicted_disease": The predicted disease name.
            - "total_history": The total number of history records.

    Raises:
        DashboardError: If any of the inputs are invalid or missing
            required data.
    """
    logger.info("Generating dashboard summary.")

    _validate_patient(patient)
    _validate_bmi_data(bmi_data)
    _validate_predicted_disease(predicted_disease)
    _validate_history(history)

    try:
        summary: dict[str, Any] = {
            "patient_name": patient.full_name,
            "age": patient.age,
            "gender": patient.gender,
            "bmi": bmi_data["bmi"],
            "bmi_category": bmi_data["category"],
            "predicted_disease": predicted_disease.strip(),
            "total_history": len(history),
        }
    except Exception as exc:
        message = "Failed to build dashboard summary from provided data."
        logger.error(message)
        raise DashboardError(message) from exc

    logger.info(
        "Dashboard summary generated successfully for patient '%s'.",
        summary["patient_name"],
    )
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    class _SamplePatient:
        """Minimal sample patient object for local testing."""

        full_name = "John Doe"
        age = 30
        gender = "Male"

    try:
        result = generate_dashboard_summary(
            patient=_SamplePatient(),
            bmi_data={"bmi": 22.86, "category": "Normal"},
            predicted_disease="Migraine",
            history=[
                {
                    "patient_name": "John Doe",
                    "disease": "Common Cold",
                    "symptoms": ["cough", "sneezing"],
                    "date": "2026-08-01",
                    "time": "10:15:00",
                }
            ],
        )
        logger.info("Dashboard summary: %s", result)
    except DashboardError:
        logger.exception("Dashboard summary generation failed.")
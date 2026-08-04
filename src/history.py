"""Diagnosis history module for the AI-Powered Healthcare Diagnosis Assistant.

This module provides reusable, framework-independent functions to create,
append, and clear diagnosis history records. History is represented as a
plain in-memory list of dictionaries; no database or external storage is
used, leaving persistence to the calling application.

Typical usage example:
    from src.history import create_history_record, append_history

    history: list[dict] = []
    record = create_history_record(
        patient_name="John Doe",
        disease="Migraine",
        symptoms=["headache", "nausea"],
    )
    history = append_history(history, record)
"""

from __future__ import annotations

import logging
from datetime import datetime

logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class HistoryError(Exception):
    """Raised when a history operation fails.

    This exception is raised whenever input validation fails while
    creating, appending, or clearing diagnosis history records.
    """


def create_history_record(
    patient_name: str,
    disease: str,
    symptoms: list[str],
) -> dict[str, str | list[str]]:
    """Create a single diagnosis history record.

    Args:
        patient_name: The name of the patient. Cannot be empty.
        disease: The diagnosed or predicted disease name. Cannot be
            empty.
        symptoms: A list of symptom names associated with the diagnosis.
            Cannot be empty.

    Returns:
        A dictionary with the following keys:
            - "patient_name": The patient's name.
            - "disease": The diagnosed disease.
            - "symptoms": The list of symptoms.
            - "date": The current date in ISO format (YYYY-MM-DD).
            - "time": The current local time (HH:MM:SS).

    Raises:
        HistoryError: If any of the inputs are invalid.
    """
    if not patient_name or not patient_name.strip():
        message = "Patient name cannot be empty."
        logger.error(message)
        raise HistoryError(message)

    if not disease or not disease.strip():
        message = "Disease cannot be empty."
        logger.error(message)
        raise HistoryError(message)

    if not isinstance(symptoms, list) or not symptoms:
        message = "Symptoms must be a non-empty list."
        logger.error(message)
        raise HistoryError(message)

    if not all(isinstance(symptom, str) and symptom.strip() for symptom in symptoms):
        message = "All symptoms must be non-empty strings."
        logger.error(message)
        raise HistoryError(message)

    now = datetime.now()
    record: dict[str, str | list[str]] = {
        "patient_name": patient_name.strip(),
        "disease": disease.strip(),
        "symptoms": list(symptoms),
        "date": now.date().isoformat(),
        "time": now.time().strftime("%H:%M:%S"),
    }

    logger.info(
        "Created history record for patient '%s' with disease '%s'.",
        record["patient_name"],
        record["disease"],
    )
    return record


def append_history(
    history: list[dict[str, str | list[str]]],
    record: dict[str, str | list[str]],
) -> list[dict[str, str | list[str]]]:
    """Append a history record to an existing history list.

    Args:
        history: The current list of history records.
        record: The history record to append. Must contain the keys
            'patient_name', 'disease', 'symptoms', 'date', and 'time'.

    Returns:
        The updated history list with the new record appended.

    Raises:
        HistoryError: If ``history`` is not a list or ``record`` is not a
            valid, well-formed history record.
    """
    if not isinstance(history, list):
        message = "History must be a list."
        logger.error(message)
        raise HistoryError(message)

    if not isinstance(record, dict):
        message = "Record must be a dictionary."
        logger.error(message)
        raise HistoryError(message)

    required_keys = {"patient_name", "disease", "symptoms", "date", "time"}
    missing_keys = required_keys - record.keys()

    if missing_keys:
        message = f"Record is missing required keys: {sorted(missing_keys)}."
        logger.error(message)
        raise HistoryError(message)

    updated_history = history + [record]

    logger.info(
        "Appended history record for patient '%s'. Total records: %d.",
        record.get("patient_name"),
        len(updated_history),
    )
    return updated_history


def clear_history(
    history: list[dict[str, str | list[str]]],
) -> list[dict[str, str | list[str]]]:
    """Clear all records from a history list.

    Args:
        history: The current list of history records.

    Returns:
        A new, empty list.

    Raises:
        HistoryError: If ``history`` is not a list.
    """
    if not isinstance(history, list):
        message = "History must be a list."
        logger.error(message)
        raise HistoryError(message)

    logger.info("Cleared history containing %d record(s).", len(history))
    return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        diagnosis_history: list[dict[str, str | list[str]]] = []

        first_record = create_history_record(
            patient_name="John Doe",
            disease="Migraine",
            symptoms=["headache", "nausea"],
        )
        diagnosis_history = append_history(diagnosis_history, first_record)
        logger.info("Current history: %s", diagnosis_history)

        second_record = create_history_record(
            patient_name="Jane Smith",
            disease="Common Cold",
            symptoms=["cough", "sneezing"],
        )
        diagnosis_history = append_history(diagnosis_history, second_record)
        logger.info("Current history: %s", diagnosis_history)

        diagnosis_history = clear_history(diagnosis_history)
        logger.info("History after clearing: %s", diagnosis_history)
    except HistoryError:
        logger.exception("History operation failed.")
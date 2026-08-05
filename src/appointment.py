"""Appointment scheduling module for the AI-Powered Healthcare Diagnosis
Assistant.

This module provides reusable, framework-independent functions to create,
append, and clear doctor appointment records. Appointments are
represented as a plain in-memory list of dictionaries; no database or
external storage is used, leaving persistence to the calling application.

Typical usage example:
    from src.appointment import create_appointment, append_appointment

    appointments: list[dict] = []
    appointment = create_appointment(
        patient_name="John Doe",
        doctor_specialization="Neurologist",
        department="Neurology",
        appointment_date="2026-08-10",
        appointment_time="10:30",
        notes="Follow-up consultation",
    )
    appointments = append_appointment(appointments, appointment)
"""

from __future__ import annotations

import logging
from datetime import datetime

logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_DATE_FORMAT: str = "%Y-%m-%d"
_TIME_FORMAT: str = "%H:%M"


class AppointmentError(Exception):
    """Raised when an appointment operation fails.

    This exception is raised whenever input validation fails while
    creating, appending, or clearing appointment records.
    """


def _validate_non_empty_string(value: str, field_name: str) -> str:
    """Validate that a value is a non-empty, stripped string.

    Args:
        value: The value to validate.
        field_name: The human-readable field name, used in error
            messages.

    Returns:
        The stripped string value.

    Raises:
        AppointmentError: If the value is not a string or is empty.
    """
    if not isinstance(value, str) or not value.strip():
        message = f"{field_name} cannot be empty."
        logger.error(message)
        raise AppointmentError(message)

    return value.strip()


def _validate_date(appointment_date: str) -> str:
    """Validate that the appointment date is in 'YYYY-MM-DD' format.

    Args:
        appointment_date: The date string to validate.

    Returns:
        The validated date string.

    Raises:
        AppointmentError: If the date is empty or improperly formatted.
    """
    date_value = _validate_non_empty_string(appointment_date, "Appointment date")

    try:
        datetime.strptime(date_value, _DATE_FORMAT)
    except ValueError as exc:
        message = (
            f"Appointment date must be in 'YYYY-MM-DD' format, "
            f"got '{appointment_date}'."
        )
        logger.error(message)
        raise AppointmentError(message) from exc

    return date_value


def _validate_time(appointment_time: str) -> str:
    """Validate that the appointment time is in 'HH:MM' 24-hour format.

    Args:
        appointment_time: The time string to validate.

    Returns:
        The validated time string.

    Raises:
        AppointmentError: If the time is empty or improperly formatted.
    """
    time_value = _validate_non_empty_string(appointment_time, "Appointment time")

    try:
        datetime.strptime(time_value, _TIME_FORMAT)
    except ValueError as exc:
        message = (
            f"Appointment time must be in 'HH:MM' 24-hour format, "
            f"got '{appointment_time}'."
        )
        logger.error(message)
        raise AppointmentError(message) from exc

    return time_value


def create_appointment(
    patient_name: str,
    doctor_specialization: str,
    department: str,
    appointment_date: str,
    appointment_time: str,
    notes: str = "",
) -> dict[str, str]:
    """Create a single appointment record.

    Args:
        patient_name: The name of the patient. Cannot be empty.
        doctor_specialization: The recommended doctor specialization.
            Cannot be empty.
        department: The relevant hospital department. Cannot be empty.
        appointment_date: The appointment date in 'YYYY-MM-DD' format.
        appointment_time: The appointment time in 'HH:MM' 24-hour format.
        notes: Optional additional notes for the appointment. Defaults to
            an empty string.

    Returns:
        A dictionary with the following keys:
            - "patient_name": The patient's name.
            - "doctor_specialization": The doctor specialization.
            - "department": The hospital department.
            - "appointment_date": The appointment date.
            - "appointment_time": The appointment time.
            - "notes": Additional notes, or an empty string.

    Raises:
        AppointmentError: If any of the required inputs are invalid.
    """
    validated_patient_name = _validate_non_empty_string(patient_name, "Patient name")
    validated_specialization = _validate_non_empty_string(
        doctor_specialization, "Doctor specialization"
    )
    validated_department = _validate_non_empty_string(department, "Department")
    validated_date = _validate_date(appointment_date)
    validated_time = _validate_time(appointment_time)

    if not isinstance(notes, str):
        message = "Notes must be a string."
        logger.error(message)
        raise AppointmentError(message)

    appointment: dict[str, str] = {
        "patient_name": validated_patient_name,
        "doctor_specialization": validated_specialization,
        "department": validated_department,
        "appointment_date": validated_date,
        "appointment_time": validated_time,
        "notes": notes.strip(),
    }

    logger.info(
        "Created appointment for patient '%s' with '%s' on %s at %s.",
        appointment["patient_name"],
        appointment["doctor_specialization"],
        appointment["appointment_date"],
        appointment["appointment_time"],
    )
    return appointment


def append_appointment(
    appointments: list[dict[str, str]],
    appointment: dict[str, str],
) -> list[dict[str, str]]:
    """Append an appointment record to an existing appointments list.

    Args:
        appointments: The current list of appointment records.
        appointment: The appointment record to append. Must contain the
            keys 'patient_name', 'doctor_specialization', 'department',
            'appointment_date', 'appointment_time', and 'notes'.

    Returns:
        The updated appointments list with the new appointment appended.

    Raises:
        AppointmentError: If ``appointments`` is not a list or
            ``appointment`` is not a valid, well-formed appointment
            record.
    """
    if not isinstance(appointments, list):
        message = "Appointments must be a list."
        logger.error(message)
        raise AppointmentError(message)

    if not isinstance(appointment, dict):
        message = "Appointment must be a dictionary."
        logger.error(message)
        raise AppointmentError(message)

    required_keys = {
        "patient_name",
        "doctor_specialization",
        "department",
        "appointment_date",
        "appointment_time",
        "notes",
    }
    missing_keys = required_keys - appointment.keys()

    if missing_keys:
        message = (
            f"Appointment is missing required keys: {sorted(missing_keys)}."
        )
        logger.error(message)
        raise AppointmentError(message)

    updated_appointments = appointments + [appointment]

    logger.info(
        "Appended appointment for patient '%s'. Total appointments: %d.",
        appointment.get("patient_name"),
        len(updated_appointments),
    )
    return updated_appointments


def clear_appointments(
    appointments: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Clear all records from an appointments list.

    Args:
        appointments: The current list of appointment records.

    Returns:
        A new, empty list.

    Raises:
        AppointmentError: If ``appointments`` is not a list.
    """
    if not isinstance(appointments, list):
        message = "Appointments must be a list."
        logger.error(message)
        raise AppointmentError(message)

    logger.info(
        "Cleared appointments list containing %d record(s).", len(appointments)
    )
    return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        appointment_list: list[dict[str, str]] = []

        first_appointment = create_appointment(
            patient_name="John Doe",
            doctor_specialization="Neurologist",
            department="Neurology",
            appointment_date="2026-08-10",
            appointment_time="10:30",
            notes="Follow-up consultation for migraine.",
        )
        appointment_list = append_appointment(appointment_list, first_appointment)
        logger.info("Current appointments: %s", appointment_list)

        second_appointment = create_appointment(
            patient_name="Jane Smith",
            doctor_specialization="General Physician",
            department="General Medicine",
            appointment_date="2026-08-12",
            appointment_time="14:00",
        )
        appointment_list = append_appointment(appointment_list, second_appointment)
        logger.info("Current appointments: %s", appointment_list)

        appointment_list = clear_appointments(appointment_list)
        logger.info("Appointments after clearing: %s", appointment_list)
    except AppointmentError:
        logger.exception("Appointment operation failed.")
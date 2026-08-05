"""Doctor recommendation module for the AI-Powered Healthcare Diagnosis
Assistant.

This module provides a reusable, framework-independent lookup of
appropriate doctor specializations, departments, urgency levels, and
general advice associated with common diseases. If a disease is not
recognized, a safe default General Physician recommendation is returned.

Typical usage example:
    from src.doctor_recommendation import get_doctor_recommendation

    recommendation = get_doctor_recommendation("Migraine")
"""

from __future__ import annotations

import logging
from typing import Final

logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DoctorRecommendationError(Exception):
    """Raised when a doctor recommendation cannot be generated.

    This exception is raised whenever the provided disease input is
    invalid (e.g., empty or not a string).
    """


_DEFAULT_RECOMMENDATION: Final[dict[str, str | bool]] = {
    "doctor_specialization": "General Physician",
    "department": "General Medicine",
    "urgency": "Routine",
    "emergency": False,
    "advice": (
        "Consult a General Physician for further evaluation, as this "
        "condition is not in our specialized database."
    ),
}

_RECOMMENDATION_DATABASE: Final[dict[str, dict[str, str | bool]]] = {
    "common cold": {
        "doctor_specialization": "General Physician",
        "department": "General Medicine",
        "urgency": "Routine",
        "emergency": False,
        "advice": "Rest, stay hydrated, and consult a physician if symptoms persist.",
    },
    "influenza": {
        "doctor_specialization": "General Physician",
        "department": "General Medicine",
        "urgency": "Routine",
        "emergency": False,
        "advice": "Consult a physician, especially if symptoms worsen or persist.",
    },
    "migraine": {
        "doctor_specialization": "Neurologist",
        "department": "Neurology",
        "urgency": "Routine",
        "emergency": False,
        "advice": "Consult a neurologist for recurring or severe headaches.",
    },
    "hypertension": {
        "doctor_specialization": "Cardiologist",
        "department": "Cardiology",
        "urgency": "Priority",
        "emergency": False,
        "advice": "Regular monitoring and consultation with a cardiologist is advised.",
    },
    "diabetes": {
        "doctor_specialization": "Endocrinologist",
        "department": "Endocrinology",
        "urgency": "Priority",
        "emergency": False,
        "advice": "Consult an endocrinologist for long-term blood sugar management.",
    },
    "asthma": {
        "doctor_specialization": "Pulmonologist",
        "department": "Pulmonology",
        "urgency": "Priority",
        "emergency": False,
        "advice": "Consult a pulmonologist, especially during frequent flare-ups.",
    },
    "gastroenteritis": {
        "doctor_specialization": "Gastroenterologist",
        "department": "Gastroenterology",
        "urgency": "Priority",
        "emergency": False,
        "advice": "Seek prompt care if dehydration or severe symptoms occur.",
    },
    "urinary tract infection": {
        "doctor_specialization": "Urologist",
        "department": "Urology",
        "urgency": "Priority",
        "emergency": False,
        "advice": "Consult a urologist or general physician for antibiotic treatment.",
    },
    "pneumonia": {
        "doctor_specialization": "Pulmonologist",
        "department": "Pulmonology",
        "urgency": "Urgent",
        "emergency": True,
        "advice": "Seek immediate medical attention, especially with breathing difficulty.",
    },
    "tuberculosis": {
        "doctor_specialization": "Pulmonologist",
        "department": "Pulmonology",
        "urgency": "Urgent",
        "emergency": False,
        "advice": "Consult a pulmonologist promptly and begin full-course treatment.",
    },
    "malaria": {
        "doctor_specialization": "Infectious Disease Specialist",
        "department": "Infectious Diseases",
        "urgency": "Urgent",
        "emergency": True,
        "advice": "Seek immediate medical attention for high fever and chills.",
    },
    "dengue": {
        "doctor_specialization": "Infectious Disease Specialist",
        "department": "Infectious Diseases",
        "urgency": "Urgent",
        "emergency": True,
        "advice": "Seek immediate care and monitor platelet count closely.",
    },
    "typhoid": {
        "doctor_specialization": "Infectious Disease Specialist",
        "department": "Infectious Diseases",
        "urgency": "Priority",
        "emergency": False,
        "advice": "Consult a physician promptly and complete the full antibiotic course.",
    },
    "chickenpox": {
        "doctor_specialization": "Dermatologist",
        "department": "Dermatology",
        "urgency": "Routine",
        "emergency": False,
        "advice": "Isolate to prevent spread and consult a physician if symptoms worsen.",
    },
    "acne": {
        "doctor_specialization": "Dermatologist",
        "department": "Dermatology",
        "urgency": "Routine",
        "emergency": False,
        "advice": "Consult a dermatologist for persistent or severe acne.",
    },
    "fungal infection": {
        "doctor_specialization": "Dermatologist",
        "department": "Dermatology",
        "urgency": "Routine",
        "emergency": False,
        "advice": "Consult a dermatologist if the infection persists or spreads.",
    },
    "gerd": {
        "doctor_specialization": "Gastroenterologist",
        "department": "Gastroenterology",
        "urgency": "Routine",
        "emergency": False,
        "advice": "Consult a gastroenterologist for persistent acid reflux symptoms.",
    },
    "arthritis": {
        "doctor_specialization": "Rheumatologist",
        "department": "Rheumatology",
        "urgency": "Routine",
        "emergency": False,
        "advice": "Consult a rheumatologist for joint pain management.",
    },
    "anemia": {
        "doctor_specialization": "Hematologist",
        "department": "Hematology",
        "urgency": "Priority",
        "emergency": False,
        "advice": "Consult a hematologist to determine the underlying cause.",
    },
    "hypothyroidism": {
        "doctor_specialization": "Endocrinologist",
        "department": "Endocrinology",
        "urgency": "Routine",
        "emergency": False,
        "advice": "Consult an endocrinologist for thyroid hormone management.",
    },
    "conjunctivitis": {
        "doctor_specialization": "Ophthalmologist",
        "department": "Ophthalmology",
        "urgency": "Routine",
        "emergency": False,
        "advice": "Consult an ophthalmologist if symptoms persist beyond a few days.",
    },
    "sinusitis": {
        "doctor_specialization": "ENT Specialist",
        "department": "Otolaryngology",
        "urgency": "Routine",
        "emergency": False,
        "advice": "Consult an ENT specialist for chronic or recurring sinusitis.",
    },
    "bronchitis": {
        "doctor_specialization": "Pulmonologist",
        "department": "Pulmonology",
        "urgency": "Priority",
        "emergency": False,
        "advice": "Consult a pulmonologist if cough persists beyond a few weeks.",
    },
}


def get_doctor_recommendation(disease: str) -> dict[str, str | bool]:
    """Retrieve a doctor recommendation for a given disease.

    This function performs a case-insensitive lookup of the provided
    disease name against an internal recommendation mapping. If the
    disease is not recognized, a safe default General Physician
    recommendation is returned instead of raising an error.

    Args:
        disease: The name of the disease to look up (case-insensitive).

    Returns:
        A dictionary with the following keys:
            - "disease": The disease name as provided (trimmed).
            - "doctor_specialization": The recommended specialist type.
            - "department": The relevant hospital department.
            - "urgency": One of 'Routine', 'Priority', or 'Urgent'.
            - "emergency": Whether the condition may require emergency
              care.
            - "advice": A short, general recommendation.

    Raises:
        DoctorRecommendationError: If the disease name is empty or not a
            string.
    """
    if not isinstance(disease, str) or not disease.strip():
        message = "Disease name must be a non-empty string."
        logger.error(message)
        raise DoctorRecommendationError(message)

    normalized_disease = disease.strip().lower()
    recommendation = _RECOMMENDATION_DATABASE.get(normalized_disease)

    if recommendation is None:
        logger.info(
            "No specific recommendation found for '%s'. Using default "
            "General Physician recommendation.",
            disease,
        )
        recommendation = _DEFAULT_RECOMMENDATION
    else:
        logger.info("Doctor recommendation retrieved for: '%s'.", disease)

    return {
        "disease": disease.strip(),
        "doctor_specialization": recommendation["doctor_specialization"],
        "department": recommendation["department"],
        "urgency": recommendation["urgency"],
        "emergency": recommendation["emergency"],
        "advice": recommendation["advice"],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        result = get_doctor_recommendation("Migraine")
        logger.info("Recommendation: %s", result)

        unknown_result = get_doctor_recommendation("Some Unknown Disease")
        logger.info("Default recommendation: %s", unknown_result)
    except DoctorRecommendationError:
        logger.exception("Doctor recommendation failed.")
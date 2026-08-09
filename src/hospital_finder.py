"""Hospital finder module for the AI-Powered Healthcare Diagnosis
Assistant.

This module provides a reusable, framework-independent lookup of
recommended hospitals based on a predicted disease, including
department, address, contact information, a Google Maps link, and
emergency availability status.

Typical usage example:
    from src.hospital_finder import get_hospital_recommendation

    recommendation = get_hospital_recommendation("Pneumonia")
"""

from __future__ import annotations

import logging
from typing import Final

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class HospitalFinderError(Exception):
    """Raised when a hospital recommendation cannot be provided.

    This exception is raised whenever the provided disease input is
    invalid (e.g., empty or not a string).
    """


_DEFAULT_RECOMMENDATION: Final[dict[str, str | bool]] = {
    "hospital_name": "City General Hospital",
    "department": "General Medicine",
    "address": "12 MG Road, Prayagraj, Uttar Pradesh, India",
    "contact_number": "+91-9999900000",
    "maps_link": "https://maps.google.com/?q=City+General+Hospital+Prayagraj",
    "emergency": False,
}

_HOSPITAL_DATABASE: Final[dict[str, dict[str, str | bool]]] = {
    "common cold": {
        "hospital_name": "City General Hospital",
        "department": "General Medicine",
        "address": "12 MG Road, Prayagraj, Uttar Pradesh, India",
        "contact_number": "+91-9999900000",
        "maps_link": "https://maps.google.com/?q=City+General+Hospital+Prayagraj",
        "emergency": False,
    },
    "influenza": {
        "hospital_name": "City General Hospital",
        "department": "General Medicine",
        "address": "12 MG Road, Prayagraj, Uttar Pradesh, India",
        "contact_number": "+91-9999900000",
        "maps_link": "https://maps.google.com/?q=City+General+Hospital+Prayagraj",
        "emergency": False,
    },
    "migraine": {
        "hospital_name": "NeuroCare Hospital",
        "department": "Neurology",
        "address": "45 Civil Lines, Prayagraj, Uttar Pradesh, India",
        "contact_number": "+91-9999900001",
        "maps_link": "https://maps.google.com/?q=NeuroCare+Hospital+Prayagraj",
        "emergency": False,
    },
    "hypertension": {
        "hospital_name": "Heartline Cardiac Institute",
        "department": "Cardiology",
        "address": "8 Tagore Town, Prayagraj, Uttar Pradesh, India",
        "contact_number": "+91-9999900002",
        "maps_link": "https://maps.google.com/?q=Heartline+Cardiac+Institute+Prayagraj",
        "emergency": True,
    },
    "diabetes": {
        "hospital_name": "MetaboCare Endocrine Center",
        "department": "Endocrinology",
        "address": "23 Katra, Prayagraj, Uttar Pradesh, India",
        "contact_number": "+91-9999900003",
        "maps_link": "https://maps.google.com/?q=MetaboCare+Endocrine+Center+Prayagraj",
        "emergency": False,
    },
    "asthma": {
        "hospital_name": "Breathe Well Pulmonary Center",
        "department": "Pulmonology",
        "address": "17 Rajapur, Prayagraj, Uttar Pradesh, India",
        "contact_number": "+91-9999900004",
        "maps_link": "https://maps.google.com/?q=Breathe+Well+Pulmonary+Center+Prayagraj",
        "emergency": True,
    },
    "gastroenteritis": {
        "hospital_name": "DigestCare Gastro Hospital",
        "department": "Gastroenterology",
        "address": "9 Allahpur, Prayagraj, Uttar Pradesh, India",
        "contact_number": "+91-9999900005",
        "maps_link": "https://maps.google.com/?q=DigestCare+Gastro+Hospital+Prayagraj",
        "emergency": True,
    },
    "urinary tract infection": {
        "hospital_name": "UroWell Hospital",
        "department": "Urology",
        "address": "31 Georgetown, Prayagraj, Uttar Pradesh, India",
        "contact_number": "+91-9999900006",
        "maps_link": "https://maps.google.com/?q=UroWell+Hospital+Prayagraj",
        "emergency": False,
    },
    "pneumonia": {
        "hospital_name": "Breathe Well Pulmonary Center",
        "department": "Pulmonology",
        "address": "17 Rajapur, Prayagraj, Uttar Pradesh, India",
        "contact_number": "+91-9999900004",
        "maps_link": "https://maps.google.com/?q=Breathe+Well+Pulmonary+Center+Prayagraj",
        "emergency": True,
    },
    "tuberculosis": {
        "hospital_name": "Breathe Well Pulmonary Center",
        "department": "Pulmonology",
        "address": "17 Rajapur, Prayagraj, Uttar Pradesh, India",
        "contact_number": "+91-9999900004",
        "maps_link": "https://maps.google.com/?q=Breathe+Well+Pulmonary+Center+Prayagraj",
        "emergency": False,
    },
    "malaria": {
        "hospital_name": "Fever Care Infectious Disease Hospital",
        "department": "Infectious Diseases",
        "address": "5 Chowk, Prayagraj, Uttar Pradesh, India",
        "contact_number": "+91-9999900007",
        "maps_link": "https://maps.google.com/?q=Fever+Care+Infectious+Disease+Hospital+Prayagraj",
        "emergency": True,
    },
    "dengue": {
        "hospital_name": "Fever Care Infectious Disease Hospital",
        "department": "Infectious Diseases",
        "address": "5 Chowk, Prayagraj, Uttar Pradesh, India",
        "contact_number": "+91-9999900007",
        "maps_link": "https://maps.google.com/?q=Fever+Care+Infectious+Disease+Hospital+Prayagraj",
        "emergency": True,
    },
    "typhoid": {
        "hospital_name": "Fever Care Infectious Disease Hospital",
        "department": "Infectious Diseases",
        "address": "5 Chowk, Prayagraj, Uttar Pradesh, India",
        "contact_number": "+91-9999900007",
        "maps_link": "https://maps.google.com/?q=Fever+Care+Infectious+Disease+Hospital+Prayagraj",
        "emergency": False,
    },
    "chickenpox": {
        "hospital_name": "SkinCare Dermatology Hospital",
        "department": "Dermatology",
        "address": "14 Mumfordganj, Prayagraj, Uttar Pradesh, India",
        "contact_number": "+91-9999900008",
        "maps_link": "https://maps.google.com/?q=SkinCare+Dermatology+Hospital+Prayagraj",
        "emergency": False,
    },
    "acne": {
        "hospital_name": "SkinCare Dermatology Hospital",
        "department": "Dermatology",
        "address": "14 Mumfordganj, Prayagraj, Uttar Pradesh, India",
        "contact_number": "+91-9999900008",
        "maps_link": "https://maps.google.com/?q=SkinCare+Dermatology+Hospital+Prayagraj",
        "emergency": False,
    },
    "fungal infection": {
        "hospital_name": "SkinCare Dermatology Hospital",
        "department": "Dermatology",
        "address": "14 Mumfordganj, Prayagraj, Uttar Pradesh, India",
        "contact_number": "+91-9999900008",
        "maps_link": "https://maps.google.com/?q=SkinCare+Dermatology+Hospital+Prayagraj",
        "emergency": False,
    },
    "gerd": {
        "hospital_name": "DigestCare Gastro Hospital",
        "department": "Gastroenterology",
        "address": "9 Allahpur, Prayagraj, Uttar Pradesh, India",
        "contact_number": "+91-9999900005",
        "maps_link": "https://maps.google.com/?q=DigestCare+Gastro+Hospital+Prayagraj",
        "emergency": False,
    },
    "arthritis": {
        "hospital_name": "JointCare Rheumatology Center",
        "department": "Rheumatology",
        "address": "27 Naini, Prayagraj, Uttar Pradesh, India",
        "contact_number": "+91-9999900009",
        "maps_link": "https://maps.google.com/?q=JointCare+Rheumatology+Center+Prayagraj",
        "emergency": False,
    },
    "anemia": {
        "hospital_name": "LifeBlood Hematology Center",
        "department": "Hematology",
        "address": "3 Colonelganj, Prayagraj, Uttar Pradesh, India",
        "contact_number": "+91-9999900010",
        "maps_link": "https://maps.google.com/?q=LifeBlood+Hematology+Center+Prayagraj",
        "emergency": False,
    },
    "hypothyroidism": {
        "hospital_name": "MetaboCare Endocrine Center",
        "department": "Endocrinology",
        "address": "23 Katra, Prayagraj, Uttar Pradesh, India",
        "contact_number": "+91-9999900003",
        "maps_link": "https://maps.google.com/?q=MetaboCare+Endocrine+Center+Prayagraj",
        "emergency": False,
    },
    "conjunctivitis": {
        "hospital_name": "ClearVision Eye Hospital",
        "department": "Ophthalmology",
        "address": "19 Zero Road, Prayagraj, Uttar Pradesh, India",
        "contact_number": "+91-9999900011",
        "maps_link": "https://maps.google.com/?q=ClearVision+Eye+Hospital+Prayagraj",
        "emergency": False,
    },
    "sinusitis": {
        "hospital_name": "EarNoseThroat Specialty Hospital",
        "department": "Otolaryngology",
        "address": "6 Kareli, Prayagraj, Uttar Pradesh, India",
        "contact_number": "+91-9999900012",
        "maps_link": "https://maps.google.com/?q=EarNoseThroat+Specialty+Hospital+Prayagraj",
        "emergency": False,
    },
    "bronchitis": {
        "hospital_name": "Breathe Well Pulmonary Center",
        "department": "Pulmonology",
        "address": "17 Rajapur, Prayagraj, Uttar Pradesh, India",
        "contact_number": "+91-9999900004",
        "maps_link": "https://maps.google.com/?q=Breathe+Well+Pulmonary+Center+Prayagraj",
        "emergency": False,
    },
}


def _validate_disease(disease: str) -> str:
    """Validate the disease input.

    Args:
        disease: The disease name to validate.

    Returns:
        The stripped disease name.

    Raises:
        HospitalFinderError: If the disease is not a non-empty string.
    """
    if not isinstance(disease, str) or not disease.strip():
        message = "Disease name must be a non-empty string."
        logger.error(message)
        raise HospitalFinderError(message)

    validated_disease = disease.strip()
    logger.info("Disease input validated: '%s'.", validated_disease)
    return validated_disease


def get_hospital_recommendation(disease: str) -> dict[str, str | bool]:
    """Retrieve a hospital recommendation for a given disease.

    This function performs a case-insensitive lookup of the provided
    disease name against an internal hospital dataset. If the disease is
    not recognized, a safe default general hospital recommendation is
    returned.

    Args:
        disease: The name of the disease to look up (case-insensitive).

    Returns:
        A dictionary with the following keys:
            - "disease": The disease name as provided (trimmed).
            - "hospital_name": The recommended hospital's name.
            - "department": The relevant hospital department.
            - "address": The hospital's address.
            - "contact_number": The hospital's contact number.
            - "maps_link": A Google Maps link to the hospital.
            - "emergency": Whether the hospital offers emergency support
              for this condition.

    Raises:
        HospitalFinderError: If the disease name is empty or not a
            string.
    """
    validated_disease = _validate_disease(disease)
    normalized_disease = validated_disease.lower()

    hospital_info = _HOSPITAL_DATABASE.get(normalized_disease)

    if hospital_info is None:
        logger.info(
            "No specific hospital recommendation found for '%s'. Using "
            "default general hospital recommendation.",
            validated_disease,
        )
        hospital_info = _DEFAULT_RECOMMENDATION
    else:
        logger.info(
            "Hospital recommendation retrieved for: '%s'.", validated_disease
        )

    recommendation: dict[str, str | bool] = {
        "disease": validated_disease,
        "hospital_name": hospital_info["hospital_name"],
        "department": hospital_info["department"],
        "address": hospital_info["address"],
        "contact_number": hospital_info["contact_number"],
        "maps_link": hospital_info["maps_link"],
        "emergency": hospital_info["emergency"],
    }

    logger.info(
        "Hospital recommendation generated successfully for disease: '%s'.",
        validated_disease,
    )
    return recommendation


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        result = get_hospital_recommendation("Pneumonia")
        logger.info("Hospital Recommendation: %s", result)

        default_result = get_hospital_recommendation("Some Unknown Disease")
        logger.info("Default Hospital Recommendation: %s", default_result)
    except HospitalFinderError:
        logger.exception("Hospital recommendation failed.")
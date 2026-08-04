"""Lab test recommendation module for the AI-Powered Healthcare Diagnosis
Assistant.

This module provides a reusable, framework-independent lookup of general,
educational information about commonly recommended laboratory tests and
their purpose for a set of common diseases.

IMPORTANT DISCLAIMER:
    The information provided by this module is for educational purposes
    only. It does NOT diagnose any disease and does NOT prescribe any
    treatment. Users should always consult a qualified healthcare
    professional for actual diagnosis and treatment decisions.

Typical usage example:
    from src.lab_tests import get_lab_test_recommendation

    recommendation = get_lab_test_recommendation("Diabetes")
"""

from __future__ import annotations

import logging
from typing import Final

logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LabTestRecommendationError(Exception):
    """Raised when a lab test recommendation cannot be provided.

    This exception is raised whenever the requested disease is not found
    in the known disease database, or the input is otherwise invalid.
    """


_DISEASE_DATABASE: Final[dict[str, dict[str, list[str]]]] = {
    "common cold": {
        "disease_name": "Common Cold",
        "recommended_tests": ["Complete Blood Count (CBC)"],
        "purpose": [
            "Rule out bacterial infection",
            "Assess white blood cell count",
        ],
    },
    "influenza": {
        "disease_name": "Influenza",
        "recommended_tests": ["Rapid Influenza Diagnostic Test", "CBC"],
        "purpose": [
            "Confirm influenza virus infection",
            "Assess severity of infection",
        ],
    },
    "migraine": {
        "disease_name": "Migraine",
        "recommended_tests": ["MRI Brain", "CT Scan"],
        "purpose": [
            "Rule out structural brain abnormalities",
            "Exclude secondary causes of headache",
        ],
    },
    "hypertension": {
        "disease_name": "Hypertension",
        "recommended_tests": [
            "Blood Pressure Monitoring",
            "Lipid Profile",
            "Kidney Function Test",
        ],
        "purpose": [
            "Confirm elevated blood pressure",
            "Assess cardiovascular risk factors",
            "Evaluate kidney function",
        ],
    },
    "diabetes": {
        "disease_name": "Diabetes",
        "recommended_tests": [
            "Fasting Blood Sugar",
            "HbA1c",
            "Random Blood Sugar",
        ],
        "purpose": [
            "Measure blood glucose",
            "Confirm diabetes",
            "Monitor long-term sugar level",
        ],
    },
    "asthma": {
        "disease_name": "Asthma",
        "recommended_tests": ["Spirometry", "Peak Flow Test", "Chest X-Ray"],
        "purpose": [
            "Assess lung function",
            "Measure airway obstruction",
            "Rule out other respiratory conditions",
        ],
    },
    "gastroenteritis": {
        "disease_name": "Gastroenteritis",
        "recommended_tests": ["Stool Culture", "Electrolyte Panel"],
        "purpose": [
            "Identify causative organism",
            "Assess dehydration and electrolyte imbalance",
        ],
    },
    "urinary tract infection": {
        "disease_name": "Urinary Tract Infection",
        "recommended_tests": ["Urinalysis", "Urine Culture"],
        "purpose": [
            "Detect presence of infection",
            "Identify causative bacteria",
            "Guide antibiotic selection",
        ],
    },
    "pneumonia": {
        "disease_name": "Pneumonia",
        "recommended_tests": ["Chest X-Ray", "CBC", "Sputum Culture"],
        "purpose": [
            "Confirm lung infection",
            "Assess severity of inflammation",
            "Identify causative organism",
        ],
    },
    "tuberculosis": {
        "disease_name": "Tuberculosis",
        "recommended_tests": [
            "Sputum Smear Microscopy",
            "Chest X-Ray",
            "GeneXpert Test",
        ],
        "purpose": [
            "Detect presence of tuberculosis bacteria",
            "Assess extent of lung involvement",
            "Confirm diagnosis rapidly",
        ],
    },
    "malaria": {
        "disease_name": "Malaria",
        "recommended_tests": ["Peripheral Blood Smear", "Rapid Diagnostic Test"],
        "purpose": [
            "Detect malaria parasites",
            "Identify parasite species",
        ],
    },
    "dengue": {
        "disease_name": "Dengue",
        "recommended_tests": ["Dengue NS1 Antigen Test", "CBC", "Platelet Count"],
        "purpose": [
            "Confirm dengue infection",
            "Monitor platelet levels",
            "Assess severity of illness",
        ],
    },
    "typhoid": {
        "disease_name": "Typhoid",
        "recommended_tests": ["Widal Test", "Blood Culture"],
        "purpose": [
            "Detect typhoid antibodies",
            "Confirm presence of Salmonella bacteria",
        ],
    },
    "chickenpox": {
        "disease_name": "Chickenpox",
        "recommended_tests": ["Clinical Skin Examination", "PCR Test"],
        "purpose": [
            "Confirm varicella-zoster virus infection",
            "Differentiate from other rash-causing conditions",
        ],
    },
    "acne": {
        "disease_name": "Acne",
        "recommended_tests": ["Hormonal Panel", "Skin Examination"],
        "purpose": [
            "Identify hormonal imbalance if present",
            "Assess severity and type of acne",
        ],
    },
    "fungal infection": {
        "disease_name": "Fungal Infection",
        "recommended_tests": ["KOH Test", "Fungal Culture"],
        "purpose": [
            "Confirm presence of fungal organisms",
            "Identify specific fungal species",
        ],
    },
    "gerd": {
        "disease_name": "GERD",
        "recommended_tests": ["Upper GI Endoscopy", "pH Monitoring"],
        "purpose": [
            "Assess esophageal damage",
            "Measure acid reflux frequency and severity",
        ],
    },
    "arthritis": {
        "disease_name": "Arthritis",
        "recommended_tests": [
            "Rheumatoid Factor Test",
            "ESR",
            "X-Ray of Joints",
        ],
        "purpose": [
            "Detect inflammatory markers",
            "Assess joint damage",
            "Differentiate arthritis type",
        ],
    },
    "anemia": {
        "disease_name": "Anemia",
        "recommended_tests": ["Complete Blood Count (CBC)", "Serum Ferritin"],
        "purpose": [
            "Measure hemoglobin levels",
            "Assess iron stores",
            "Determine cause of anemia",
        ],
    },
    "hypothyroidism": {
        "disease_name": "Hypothyroidism",
        "recommended_tests": ["TSH Test", "Free T4 Test"],
        "purpose": [
            "Assess thyroid hormone levels",
            "Confirm underactive thyroid function",
        ],
    },
    "conjunctivitis": {
        "disease_name": "Conjunctivitis",
        "recommended_tests": ["Eye Swab Culture", "Clinical Examination"],
        "purpose": [
            "Identify causative organism",
            "Differentiate bacterial from viral cause",
        ],
    },
    "sinusitis": {
        "disease_name": "Sinusitis",
        "recommended_tests": ["Sinus X-Ray", "CT Scan of Sinuses"],
        "purpose": [
            "Assess sinus inflammation",
            "Identify blockages or structural issues",
        ],
    },
    "bronchitis": {
        "disease_name": "Bronchitis",
        "recommended_tests": ["Chest X-Ray", "Sputum Test", "Spirometry"],
        "purpose": [
            "Rule out pneumonia",
            "Identify causative organism",
            "Assess lung function",
        ],
    },
}


def get_lab_test_recommendation(disease: str) -> dict[str, list[str] | str]:
    """Retrieve educational lab test recommendation information for a disease.

    This function performs a case-insensitive lookup of the provided
    disease name against a curated database of common diseases and
    returns general, educational information about commonly recommended
    laboratory tests and their purpose.

    Args:
        disease: The name of the disease to look up (case-insensitive).

    Returns:
        A dictionary with the following keys:
            - "disease": The canonical disease name.
            - "recommended_tests": A list of commonly recommended
              laboratory tests.
            - "purpose": A list describing the purpose of the tests.

    Raises:
        LabTestRecommendationError: If the disease name is empty or not
            found in the database.

    Note:
        This information is provided for educational purposes only. It
        does not diagnose any disease and does not prescribe any
        treatment. Always consult a qualified healthcare professional.
    """
    if not disease or not disease.strip():
        message = "Disease name cannot be empty."
        logger.error(message)
        raise LabTestRecommendationError(message)

    normalized_disease = disease.strip().lower()
    disease_info = _DISEASE_DATABASE.get(normalized_disease)

    if disease_info is None:
        message = (
            f"No lab test recommendation found for disease: '{disease}'."
        )
        logger.error(message)
        raise LabTestRecommendationError(message)

    logger.info("Lab test recommendation retrieved for: '%s'.", disease)

    return {
        "disease": disease_info["disease_name"],
        "recommended_tests": list(disease_info["recommended_tests"]),
        "purpose": list(disease_info["purpose"]),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        result = get_lab_test_recommendation("Diabetes")
        logger.info("Disease: %s", result["disease"])
        logger.info("Recommended Tests: %s", result["recommended_tests"])
        logger.info("Purpose: %s", result["purpose"])
    except LabTestRecommendationError:
        logger.exception("Lab test recommendation failed.")
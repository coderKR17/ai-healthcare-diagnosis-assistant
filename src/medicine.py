"""Medicine recommendation module for the AI-Powered Healthcare Diagnosis
Assistant.

This module provides a reusable, framework-independent lookup of general,
educational information about commonly associated over-the-counter
medicine categories and precautions for a set of common diseases.

IMPORTANT DISCLAIMER:
    The information provided by this module is for educational purposes
    only. It does NOT include dosage information and does NOT constitute
    a medical prescription or treatment plan. Users should always consult
    a qualified healthcare professional before taking any medication.

Typical usage example:
    from src.medicine import get_medicine_recommendation

    recommendation = get_medicine_recommendation("Migraine")
"""

from __future__ import annotations

import logging
from typing import Final

logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class MedicineRecommendationError(Exception):
    """Raised when a medicine recommendation cannot be provided.

    This exception is raised whenever the requested disease is not found
    in the known disease database, or the input is otherwise invalid.
    """


_DISEASE_DATABASE: Final[dict[str, dict[str, list[str]]]] = {
    "common cold": {
        "disease_name": "Common Cold",
        "common_medicines": ["Antihistamines", "Decongestants", "Antipyretics"],
        "precautions": [
            "Stay hydrated",
            "Get adequate rest",
            "Avoid cold beverages",
            "Cover mouth while coughing or sneezing",
        ],
    },
    "influenza": {
        "disease_name": "Influenza",
        "common_medicines": ["Antivirals", "Antipyretics", "Cough suppressants"],
        "precautions": [
            "Isolate from others to prevent spread",
            "Stay hydrated",
            "Get plenty of rest",
            "Wash hands frequently",
        ],
    },
    "migraine": {
        "disease_name": "Migraine",
        "common_medicines": ["Analgesics", "Triptans", "Anti-nausea medication"],
        "precautions": [
            "Avoid known triggers such as bright lights",
            "Maintain a regular sleep schedule",
            "Stay hydrated",
            "Rest in a quiet, dark room during attacks",
        ],
    },
    "hypertension": {
        "disease_name": "Hypertension",
        "common_medicines": [
            "ACE inhibitors",
            "Beta blockers",
            "Diuretics",
            "Calcium channel blockers",
        ],
        "precautions": [
            "Reduce salt intake",
            "Exercise regularly",
            "Monitor blood pressure regularly",
            "Limit alcohol consumption",
        ],
    },
    "diabetes": {
        "disease_name": "Diabetes",
        "common_medicines": ["Metformin", "Insulin", "Sulfonylureas"],
        "precautions": [
            "Monitor blood sugar levels regularly",
            "Maintain a balanced, low-sugar diet",
            "Exercise regularly",
            "Attend routine checkups",
        ],
    },
    "asthma": {
        "disease_name": "Asthma",
        "common_medicines": ["Bronchodilators", "Inhaled corticosteroids"],
        "precautions": [
            "Avoid known triggers such as dust and smoke",
            "Keep a rescue inhaler accessible",
            "Monitor breathing regularly",
            "Avoid strenuous activity during flare-ups",
        ],
    },
    "gastroenteritis": {
        "disease_name": "Gastroenteritis",
        "common_medicines": ["Oral rehydration salts", "Antiemetics", "Antidiarrheals"],
        "precautions": [
            "Stay hydrated",
            "Eat bland, easy-to-digest food",
            "Avoid dairy and fatty foods",
            "Practice good hand hygiene",
        ],
    },
    "urinary tract infection": {
        "disease_name": "Urinary Tract Infection",
        "common_medicines": ["Antibiotics", "Analgesics for urinary discomfort"],
        "precautions": [
            "Drink plenty of water",
            "Urinate frequently and avoid holding urine",
            "Maintain proper hygiene",
            "Avoid irritating feminine products",
        ],
    },
    "pneumonia": {
        "disease_name": "Pneumonia",
        "common_medicines": ["Antibiotics", "Antipyretics", "Cough medicine"],
        "precautions": [
            "Get plenty of rest",
            "Stay hydrated",
            "Avoid smoking",
            "Seek immediate care if breathing worsens",
        ],
    },
    "tuberculosis": {
        "disease_name": "Tuberculosis",
        "common_medicines": [
            "Isoniazid",
            "Rifampicin",
            "Ethambutol",
            "Pyrazinamide",
        ],
        "precautions": [
            "Complete the full course of treatment",
            "Isolate during the infectious period",
            "Cover mouth while coughing",
            "Ensure good ventilation in living areas",
        ],
    },
    "malaria": {
        "disease_name": "Malaria",
        "common_medicines": ["Antimalarials", "Antipyretics"],
        "precautions": [
            "Use mosquito nets and repellents",
            "Stay hydrated",
            "Get plenty of rest",
            "Seek immediate medical attention for high fever",
        ],
    },
    "dengue": {
        "disease_name": "Dengue",
        "common_medicines": ["Antipyretics (avoiding NSAIDs)"],
        "precautions": [
            "Stay hydrated",
            "Monitor platelet count regularly",
            "Avoid mosquito bites",
            "Seek immediate care for warning signs like bleeding",
        ],
    },
    "typhoid": {
        "disease_name": "Typhoid",
        "common_medicines": ["Antibiotics", "Antipyretics"],
        "precautions": [
            "Drink safe, clean water",
            "Maintain proper food hygiene",
            "Get adequate rest",
            "Complete the full course of antibiotics",
        ],
    },
    "chickenpox": {
        "disease_name": "Chickenpox",
        "common_medicines": ["Antihistamines", "Antipyretics", "Calamine lotion"],
        "precautions": [
            "Avoid scratching the rash",
            "Isolate to prevent spreading",
            "Keep skin clean and dry",
            "Trim fingernails to reduce skin damage",
        ],
    },
    "acne": {
        "disease_name": "Acne",
        "common_medicines": ["Topical retinoids", "Benzoyl peroxide", "Antibiotics"],
        "precautions": [
            "Avoid touching or picking at the skin",
            "Use non-comedogenic skincare products",
            "Wash face gently twice daily",
            "Avoid excessive sun exposure",
        ],
    },
    "fungal infection": {
        "disease_name": "Fungal Infection",
        "common_medicines": ["Antifungal creams", "Oral antifungals"],
        "precautions": [
            "Keep the affected area clean and dry",
            "Avoid sharing personal items",
            "Wear breathable clothing",
            "Avoid prolonged moisture on skin",
        ],
    },
    "gerd": {
        "disease_name": "GERD",
        "common_medicines": ["Antacids", "Proton pump inhibitors", "H2 blockers"],
        "precautions": [
            "Avoid spicy and fatty foods",
            "Avoid lying down immediately after eating",
            "Maintain a healthy weight",
            "Limit caffeine and alcohol intake",
        ],
    },
    "arthritis": {
        "disease_name": "Arthritis",
        "common_medicines": ["NSAIDs", "Analgesics", "Corticosteroids"],
        "precautions": [
            "Engage in low-impact exercise",
            "Maintain a healthy weight",
            "Apply hot or cold therapy as needed",
            "Avoid repetitive joint strain",
        ],
    },
    "anemia": {
        "disease_name": "Anemia",
        "common_medicines": ["Iron supplements", "Vitamin B12 supplements"],
        "precautions": [
            "Eat iron-rich foods",
            "Include vitamin C to aid iron absorption",
            "Get adequate rest",
            "Attend routine blood tests",
        ],
    },
    "hypothyroidism": {
        "disease_name": "Hypothyroidism",
        "common_medicines": ["Levothyroxine"],
        "precautions": [
            "Take medication consistently as prescribed",
            "Attend regular thyroid function tests",
            "Maintain a balanced diet",
            "Report symptoms of fatigue or weight changes",
        ],
    },
    "conjunctivitis": {
        "disease_name": "Conjunctivitis",
        "common_medicines": ["Antibiotic eye drops", "Antihistamine eye drops"],
        "precautions": [
            "Avoid touching or rubbing the eyes",
            "Wash hands frequently",
            "Avoid sharing towels or pillows",
            "Discard contact lenses if applicable",
        ],
    },
    "sinusitis": {
        "disease_name": "Sinusitis",
        "common_medicines": ["Decongestants", "Nasal corticosteroid sprays"],
        "precautions": [
            "Use steam inhalation for relief",
            "Stay hydrated",
            "Avoid allergens and irritants",
            "Rest adequately",
        ],
    },
    "bronchitis": {
        "disease_name": "Bronchitis",
        "common_medicines": ["Cough suppressants", "Bronchodilators", "Antipyretics"],
        "precautions": [
            "Avoid smoking and secondhand smoke",
            "Stay hydrated",
            "Use a humidifier",
            "Get plenty of rest",
        ],
    },
}


def get_medicine_recommendation(disease: str) -> dict[str, list[str] | str]:
    """Retrieve educational medicine and precaution information for a disease.

    This function performs a case-insensitive lookup of the provided
    disease name against a curated database of common diseases and
    returns general, educational information about commonly associated
    medicine categories and precautions.

    Args:
        disease: The name of the disease to look up (case-insensitive).

    Returns:
        A dictionary with the following keys:
            - "disease": The canonical disease name.
            - "common_medicines": A list of commonly associated medicine
              categories.
            - "precautions": A list of recommended precautions.

    Raises:
        MedicineRecommendationError: If the disease name is empty or not
            found in the database.

    Note:
        This information is provided for educational purposes only. It
        does not include dosage information and does not constitute a
        medical prescription or treatment plan. Always consult a
        qualified healthcare professional.
    """
    if not disease or not disease.strip():
        message = "Disease name cannot be empty."
        logger.error(message)
        raise MedicineRecommendationError(message)

    normalized_disease = disease.strip().lower()
    disease_info = _DISEASE_DATABASE.get(normalized_disease)

    if disease_info is None:
        message = f"No medicine recommendation found for disease: '{disease}'."
        logger.error(message)
        raise MedicineRecommendationError(message)

    logger.info("Medicine recommendation retrieved for: '%s'.", disease)

    return {
        "disease": disease_info["disease_name"],
        "common_medicines": list(disease_info["common_medicines"]),
        "precautions": list(disease_info["precautions"]),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        result = get_medicine_recommendation("Migraine")
        logger.info("Disease: %s", result["disease"])
        logger.info("Common Medicines: %s", result["common_medicines"])
        logger.info("Precautions: %s", result["precautions"])
    except MedicineRecommendationError:
        logger.exception("Medicine recommendation failed.")
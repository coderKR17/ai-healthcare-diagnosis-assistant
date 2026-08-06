"""Health tips module for the AI-Powered Healthcare Diagnosis Assistant.

This module provides a reusable, framework-independent function to
generate personalized diet, exercise, sleep, hydration, and lifestyle
recommendations based on a predicted disease and BMI category. If the
disease is not recognized, safe general health advice is returned
instead of raising an error.

Typical usage example:
    from src.health_tips import get_health_tips

    tips = get_health_tips(disease="Diabetes", bmi_category="Overweight")
"""

from __future__ import annotations

import logging
from typing import Final

logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

VALID_BMI_CATEGORIES: Final[tuple[str, ...]] = (
    "Underweight",
    "Normal",
    "Overweight",
    "Obese",
)


class HealthTipsError(Exception):
    """Raised when health tips cannot be generated.

    This exception is raised whenever the provided disease or BMI
    category input is invalid.
    """


_DEFAULT_TIPS: Final[dict[str, list[str] | str]] = {
    "diet": [
        "Eat a balanced diet rich in fruits, vegetables, and whole grains.",
        "Limit processed foods, added sugar, and excess salt.",
    ],
    "exercise": [
        "Engage in at least 30 minutes of moderate physical activity most days.",
    ],
    "sleep": "Aim for 7-9 hours of quality sleep each night.",
    "hydration": "Drink at least 8 glasses of water daily, unless advised otherwise.",
    "lifestyle": [
        "Avoid smoking and limit alcohol consumption.",
        "Manage stress through relaxation techniques or hobbies.",
    ],
    "additional_tips": [
        "Schedule regular health checkups with your physician.",
    ],
}

_DISEASE_TIPS: Final[dict[str, dict[str, list[str] | str]]] = {
    "diabetes": {
        "diet": [
            "Choose low-glycemic-index foods such as whole grains and legumes.",
            "Limit intake of sugary foods and refined carbohydrates.",
        ],
        "exercise": [
            "Engage in regular aerobic activity such as brisk walking.",
            "Include light strength training two to three times a week.",
        ],
        "sleep": "Maintain a consistent sleep schedule of 7-8 hours nightly.",
        "hydration": "Drink adequate water and avoid sugary beverages.",
        "lifestyle": [
            "Monitor blood sugar levels regularly.",
            "Avoid smoking, as it worsens insulin resistance.",
        ],
        "additional_tips": [
            "Work with a dietitian to plan balanced, portion-controlled meals.",
        ],
    },
    "hypertension": {
        "diet": [
            "Reduce sodium intake and avoid processed or salty foods.",
            "Increase potassium-rich foods such as bananas and leafy greens.",
        ],
        "exercise": [
            "Engage in moderate aerobic exercise like walking or cycling.",
        ],
        "sleep": "Aim for 7-8 hours of restful sleep to support healthy blood pressure.",
        "hydration": "Stay well hydrated with water throughout the day.",
        "lifestyle": [
            "Limit alcohol consumption and avoid smoking.",
            "Practice stress-reduction techniques such as meditation.",
        ],
        "additional_tips": [
            "Monitor blood pressure regularly and track trends over time.",
        ],
    },
    "asthma": {
        "diet": [
            "Include anti-inflammatory foods such as fruits and omega-3-rich fish.",
        ],
        "exercise": [
            "Engage in light to moderate exercise with a pre-approved plan.",
        ],
        "sleep": "Ensure a clean, allergen-free sleeping environment.",
        "hydration": "Stay hydrated to help keep airway mucus thin.",
        "lifestyle": [
            "Avoid known triggers such as smoke, dust, and strong odors.",
            "Keep a rescue inhaler accessible at all times.",
        ],
        "additional_tips": [
            "Track symptoms and triggers in a daily journal.",
        ],
    },
    "gerd": {
        "diet": [
            "Avoid spicy, fatty, and acidic foods.",
            "Eat smaller, more frequent meals instead of large ones.",
        ],
        "exercise": [
            "Avoid vigorous exercise immediately after eating.",
        ],
        "sleep": "Avoid lying down within 2-3 hours after meals.",
        "hydration": "Drink water between meals rather than during them.",
        "lifestyle": [
            "Maintain a healthy weight to reduce abdominal pressure.",
            "Avoid smoking and excessive alcohol consumption.",
        ],
        "additional_tips": [
            "Elevate the head of the bed to reduce nighttime reflux.",
        ],
    },
    "arthritis": {
        "diet": [
            "Include anti-inflammatory foods such as fatty fish and nuts.",
        ],
        "exercise": [
            "Engage in low-impact exercises such as swimming or yoga.",
        ],
        "sleep": "Maintain consistent sleep to help manage pain and inflammation.",
        "hydration": "Stay well hydrated to support joint lubrication.",
        "lifestyle": [
            "Maintain a healthy weight to reduce joint strain.",
            "Use joint protection techniques during daily activities.",
        ],
        "additional_tips": [
            "Apply hot or cold therapy as needed for pain relief.",
        ],
    },
    "anemia": {
        "diet": [
            "Include iron-rich foods such as leafy greens, beans, and lean meat.",
            "Pair iron-rich foods with vitamin C to improve absorption.",
        ],
        "exercise": [
            "Engage in light to moderate exercise as tolerated.",
        ],
        "sleep": "Ensure adequate rest to help the body recover energy levels.",
        "hydration": "Stay adequately hydrated throughout the day.",
        "lifestyle": [
            "Avoid excessive tea or coffee with meals, as they can hinder iron absorption.",
        ],
        "additional_tips": [
            "Follow up with routine blood tests to monitor iron levels.",
        ],
    },
    "common cold": {
        "diet": [
            "Consume warm fluids such as soups and herbal teas.",
            "Include vitamin C-rich foods like citrus fruits.",
        ],
        "exercise": [
            "Rest and avoid strenuous exercise until symptoms resolve.",
        ],
        "sleep": "Prioritize extra rest and sleep to support recovery.",
        "hydration": "Increase fluid intake to stay well hydrated.",
        "lifestyle": [
            "Wash hands frequently to prevent spreading infection.",
        ],
        "additional_tips": [
            "Use a humidifier to ease nasal congestion.",
        ],
    },
}


def _validate_disease(disease: str) -> str:
    """Validate the disease input.

    Args:
        disease: The disease name to validate.

    Returns:
        The stripped, lowercase disease name.

    Raises:
        HealthTipsError: If the disease is not a non-empty string.
    """
    if not isinstance(disease, str) or not disease.strip():
        message = "Disease name must be a non-empty string."
        logger.error(message)
        raise HealthTipsError(message)

    return disease.strip()


def _validate_bmi_category(bmi_category: str) -> str:
    """Validate the BMI category input.

    Args:
        bmi_category: The BMI category to validate.

    Returns:
        The validated BMI category.

    Raises:
        HealthTipsError: If the BMI category is invalid.
    """
    if not isinstance(bmi_category, str) or not bmi_category.strip():
        message = "BMI category must be a non-empty string."
        logger.error(message)
        raise HealthTipsError(message)

    normalized_category = bmi_category.strip()

    if normalized_category not in VALID_BMI_CATEGORIES:
        message = (
            f"BMI category must be one of {VALID_BMI_CATEGORIES}, "
            f"got '{bmi_category}'."
        )
        logger.error(message)
        raise HealthTipsError(message)

    return normalized_category


def _get_bmi_personalization(bmi_category: str) -> list[str]:
    """Generate BMI-specific lifestyle recommendations.

    Args:
        bmi_category: A validated BMI category.

    Returns:
        A list of personalized lifestyle tips based on the BMI category.
    """
    bmi_tips: dict[str, list[str]] = {
        "Underweight": [
            "Increase caloric intake with nutrient-dense, healthy foods.",
            "Include protein-rich foods to support muscle growth.",
        ],
        "Normal": [
            "Maintain your current healthy weight through balanced habits.",
        ],
        "Overweight": [
            "Focus on portion control and regular physical activity.",
            "Gradually reduce calorie-dense and processed foods.",
        ],
        "Obese": [
            "Consult a healthcare provider for a structured weight management plan.",
            "Incorporate consistent, sustainable physical activity.",
        ],
    }
    return bmi_tips.get(bmi_category, [])


def get_health_tips(disease: str, bmi_category: str) -> dict[str, list[str] | str]:
    """Generate personalized health tips based on disease and BMI category.

    This function performs a case-insensitive lookup of the provided
    disease name against an internal recommendation mapping and
    personalizes the lifestyle recommendations using the given BMI
    category. If the disease is not recognized, safe general health
    advice is returned instead of raising an error.

    Args:
        disease: The name of the disease to generate tips for
            (case-insensitive).
        bmi_category: The patient's BMI category. Must be one of
            'Underweight', 'Normal', 'Overweight', or 'Obese'.

    Returns:
        A dictionary with the following keys:
            - "diet": A list of dietary recommendations.
            - "exercise": A list of exercise recommendations.
            - "sleep": A sleep recommendation string.
            - "hydration": A hydration recommendation string.
            - "lifestyle": A list of lifestyle recommendations,
              personalized using the BMI category.
            - "additional_tips": A list of additional general tips.

    Raises:
        HealthTipsError: If the disease name or BMI category is invalid.
    """
    validated_disease = _validate_disease(disease)
    validated_bmi_category = _validate_bmi_category(bmi_category)

    normalized_disease = validated_disease.lower()
    disease_tips = _DISEASE_TIPS.get(normalized_disease)

    if disease_tips is None:
        logger.info(
            "No specific health tips found for '%s'. Using default "
            "general health advice.",
            validated_disease,
        )
        disease_tips = _DEFAULT_TIPS
    else:
        logger.info("Health tips retrieved for disease: '%s'.", validated_disease)

    bmi_personalization = _get_bmi_personalization(validated_bmi_category)
    combined_lifestyle = list(disease_tips["lifestyle"]) + bmi_personalization

    result: dict[str, list[str] | str] = {
        "diet": list(disease_tips["diet"]),
        "exercise": list(disease_tips["exercise"]),
        "sleep": disease_tips["sleep"],
        "hydration": disease_tips["hydration"],
        "lifestyle": combined_lifestyle,
        "additional_tips": list(disease_tips["additional_tips"]),
    }

    logger.info(
        "Health tips generated for disease '%s' with BMI category '%s'.",
        validated_disease,
        validated_bmi_category,
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        result = get_health_tips(disease="Diabetes", bmi_category="Overweight")
        logger.info("Health Tips: %s", result)

        default_result = get_health_tips(
            disease="Some Unknown Disease", bmi_category="Normal"
        )
        logger.info("Default Health Tips: %s", default_result)
    except HealthTipsError:
        logger.exception("Health tips generation failed.")
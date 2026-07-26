"""BMI calculation module for the AI-Powered Healthcare Diagnosis Assistant.

This module provides reusable, framework-independent functions to
calculate Body Mass Index (BMI), determine the corresponding WHO BMI
category, assess the associated health risk level, and provide a
professional health recommendation based on that category.

Typical usage example:
    from src.bmi import (
        calculate_bmi,
        get_bmi_category,
        get_health_risk,
        get_health_tip,
    )

    bmi = calculate_bmi(height_cm=175.0, weight_kg=70.0)
    category = get_bmi_category(bmi)
    risk = get_health_risk(bmi)
    tip = get_health_tip(category)
"""

from __future__ import annotations

import logging
from typing import Final

logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

MIN_HEIGHT_CM: Final[float] = 50.0
MAX_HEIGHT_CM: Final[float] = 250.0
MIN_WEIGHT_KG: Final[float] = 2.0
MAX_WEIGHT_KG: Final[float] = 300.0

UNDERWEIGHT_THRESHOLD: Final[float] = 18.5
NORMAL_THRESHOLD: Final[float] = 25.0
OVERWEIGHT_THRESHOLD: Final[float] = 30.0

CATEGORY_UNDERWEIGHT: Final[str] = "Underweight"
CATEGORY_NORMAL: Final[str] = "Normal"
CATEGORY_OVERWEIGHT: Final[str] = "Overweight"
CATEGORY_OBESE: Final[str] = "Obese"

RISK_LOW: Final[str] = "Low Risk"
RISK_MODERATE: Final[str] = "Moderate Risk"
RISK_HIGH: Final[str] = "High Risk"
RISK_VERY_HIGH: Final[str] = "Very High Risk"

_HEALTH_TIPS: Final[dict[str, str]] = {
    CATEGORY_UNDERWEIGHT: (
        "Consider increasing your caloric intake with nutrient-dense foods "
        "and consult a healthcare provider or dietitian to address "
        "potential underlying causes of low body weight."
    ),
    CATEGORY_NORMAL: (
        "Maintain your current weight through a balanced diet and regular "
        "physical activity, and continue routine health checkups."
    ),
    CATEGORY_OVERWEIGHT: (
        "Adopt a balanced, calorie-controlled diet and increase physical "
        "activity gradually; consult a healthcare provider for a "
        "personalized weight management plan."
    ),
    CATEGORY_OBESE: (
        "Consult a healthcare provider promptly to develop a comprehensive "
        "weight management plan, as obesity increases the risk of several "
        "chronic conditions."
    ),
}


class BMICalculationError(Exception):
    """Raised when BMI calculation or classification fails.

    This exception is raised whenever height or weight values are invalid,
    or when a BMI-related computation cannot be completed successfully.
    """


def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    """Calculate the Body Mass Index (BMI) from height and weight.

    Args:
        height_cm: Height in centimeters. Must be between 50 and 250.
        weight_kg: Weight in kilograms. Must be between 2 and 300.

    Returns:
        The BMI value rounded to 2 decimal places.

    Raises:
        BMICalculationError: If height or weight is out of the valid
            range, or if the BMI cannot be computed.
    """
    if not (MIN_HEIGHT_CM <= height_cm <= MAX_HEIGHT_CM):
        message = (
            f"Height must be between {MIN_HEIGHT_CM} and {MAX_HEIGHT_CM} "
            f"cm, got {height_cm}."
        )
        logger.error(message)
        raise BMICalculationError(message)

    if not (MIN_WEIGHT_KG <= weight_kg <= MAX_WEIGHT_KG):
        message = (
            f"Weight must be between {MIN_WEIGHT_KG} and {MAX_WEIGHT_KG} "
            f"kg, got {weight_kg}."
        )
        logger.error(message)
        raise BMICalculationError(message)

    try:
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
    except (ZeroDivisionError, ArithmeticError) as exc:
        message = "Failed to calculate BMI due to invalid measurements."
        logger.error(message)
        raise BMICalculationError(message) from exc

    rounded_bmi = round(bmi, 2)
    logger.info(
        "Calculated BMI: %.2f (height=%.2f cm, weight=%.2f kg)",
        rounded_bmi,
        height_cm,
        weight_kg,
    )
    return rounded_bmi


def get_bmi_category(bmi: float) -> str:
    """Determine the WHO BMI category for a given BMI value.

    Args:
        bmi: The BMI value to classify. Must be a positive number.

    Returns:
        One of 'Underweight', 'Normal', 'Overweight', or 'Obese'.

    Raises:
        BMICalculationError: If the BMI value is invalid (not positive).
    """
    if bmi <= 0:
        message = f"BMI must be a positive number, got {bmi}."
        logger.error(message)
        raise BMICalculationError(message)

    if bmi < UNDERWEIGHT_THRESHOLD:
        category = CATEGORY_UNDERWEIGHT
    elif bmi < NORMAL_THRESHOLD:
        category = CATEGORY_NORMAL
    elif bmi < OVERWEIGHT_THRESHOLD:
        category = CATEGORY_OVERWEIGHT
    else:
        category = CATEGORY_OBESE

    logger.info("BMI %.2f classified as category: %s", bmi, category)
    return category


def get_health_risk(bmi: float) -> str:
    """Determine the health risk level associated with a given BMI value.

    Args:
        bmi: The BMI value to assess. Must be a positive number.

    Returns:
        One of 'Low Risk', 'Moderate Risk', 'High Risk', or
        'Very High Risk'.

    Raises:
        BMICalculationError: If the BMI value is invalid (not positive).
    """
    if bmi <= 0:
        message = f"BMI must be a positive number, got {bmi}."
        logger.error(message)
        raise BMICalculationError(message)

    if bmi < UNDERWEIGHT_THRESHOLD:
        risk = RISK_MODERATE
    elif bmi < NORMAL_THRESHOLD:
        risk = RISK_LOW
    elif bmi < OVERWEIGHT_THRESHOLD:
        risk = RISK_MODERATE
    elif bmi < 35:
        risk = RISK_HIGH
    else:
        risk = RISK_VERY_HIGH

    logger.info("BMI %.2f classified as health risk: %s", bmi, risk)
    return risk


def get_health_tip(category: str) -> str:
    """Provide a professional health recommendation for a BMI category.

    Args:
        category: The BMI category. Must be one of 'Underweight',
            'Normal', 'Overweight', or 'Obese'.

    Returns:
        A professional health recommendation string corresponding to the
        given category.

    Raises:
        BMICalculationError: If the category is not recognized.
    """
    tip = _HEALTH_TIPS.get(category)

    if tip is None:
        message = (
            f"Unrecognized BMI category: '{category}'. Expected one of "
            f"{list(_HEALTH_TIPS.keys())}."
        )
        logger.error(message)
        raise BMICalculationError(message)

    logger.info("Retrieved health tip for category: %s", category)
    return tip


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        sample_bmi = calculate_bmi(height_cm=175.0, weight_kg=70.0)
        sample_category = get_bmi_category(sample_bmi)
        sample_risk = get_health_risk(sample_bmi)
        sample_tip = get_health_tip(sample_category)

        logger.info("BMI: %.2f", sample_bmi)
        logger.info("Category: %s", sample_category)
        logger.info("Health Risk: %s", sample_risk)
        logger.info("Health Tip: %s", sample_tip)
    except BMICalculationError:
        logger.exception("BMI calculation failed.")
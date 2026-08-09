"""AI Health Chatbot module for the AI-Powered Healthcare Diagnosis
Assistant.

This module provides a reusable, framework-independent chatbot that
answers general health-related questions. It uses a safe, rule-based
response system by default and can optionally leverage an external AI
provider if credentials are supplied via environment variables. If no
API key is available, or if the API call fails for any reason, the
chatbot gracefully falls back to the rule-based system.

The chatbot never diagnoses disease, never prescribes medicines or
dosages, and never replaces professional medical consultation.

Typical usage example:
    from src.health_chatbot import get_chatbot_response

    reply = get_chatbot_response(
        user_message="What should I eat for diabetes?",
        predicted_disease="Diabetes",
        bmi=24.5,
    )
"""

from __future__ import annotations

import logging
import os
import re
from typing import Final

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

MEDICAL_DISCLAIMER: Final[str] = (
    "This information is for general educational purposes only and is "
    "not a medical diagnosis or prescription. Please consult a "
    "qualified healthcare professional for personalized advice."
)

EMERGENCY_DISCLAIMER: Final[str] = (
    "⚠️ This may be a medical emergency. Please seek immediate "
    "professional medical care or contact your local emergency services "
    "right away."
)

AI_API_KEY_ENV_VAR: Final[str] = "HEALTH_CHATBOT_API_KEY"
AI_API_PROVIDER_ENV_VAR: Final[str] = "HEALTH_CHATBOT_API_PROVIDER"


class HealthChatbotError(Exception):
    """Raised when the health chatbot cannot generate a response.

    This exception is raised whenever the provided user message,
    predicted disease, or BMI input fails validation.
    """


_EMERGENCY_KEYWORDS: Final[tuple[str, ...]] = (
    "chest pain",
    "difficulty breathing",
    "can't breathe",
    "cannot breathe",
    "severe bleeding",
    "unconscious",
    "unresponsive",
    "stroke",
    "heart attack",
    "seizure",
    "suicidal",
    "suicide",
    "severe allergic reaction",
    "anaphylaxis",
    "choking",
    "not breathing",
)

_TOPIC_KEYWORDS: Final[dict[str, tuple[str, ...]]] = {
    "diabetes": ("diabetes", "blood sugar", "glucose"),
    "hypertension": ("hypertension", "blood pressure", "bp"),
    "fever": ("fever", "temperature"),
    "common_cold": ("cold", "runny nose", "sneezing"),
    "cough": ("cough",),
    "headache": ("headache", "migraine"),
    "bmi": ("bmi", "body mass index"),
    "diet": ("diet", "food", "nutrition", "eating"),
    "exercise": ("exercise", "workout", "physical activity", "gym"),
    "sleep": ("sleep", "insomnia", "rest"),
    "hydration": ("water", "hydration", "hydrate", "fluids"),
    "lifestyle": ("lifestyle", "habits", "smoking", "alcohol"),
    "medicine": ("medicine", "medication", "drug", "dosage", "pill", "tablet"),
    "doctor": ("doctor", "physician", "specialist", "consult"),
}

_TOPIC_RESPONSES: Final[dict[str, str]] = {
    "diabetes": (
        "Diabetes management typically involves monitoring blood sugar "
        "levels, maintaining a balanced low-sugar diet, staying "
        "physically active, and taking medications as prescribed by "
        "your doctor."
    ),
    "hypertension": (
        "Managing high blood pressure generally involves reducing salt "
        "intake, staying physically active, managing stress, and "
        "regularly monitoring your blood pressure."
    ),
    "fever": (
        "A fever is often the body's response to infection. Rest, stay "
        "hydrated, and monitor your temperature. Seek medical care if "
        "the fever is high, persistent, or accompanied by other "
        "concerning symptoms."
    ),
    "common_cold": (
        "For a common cold, rest, stay hydrated, and consider warm "
        "fluids. Most colds resolve on their own within a week to ten "
        "days."
    ),
    "cough": (
        "A cough can result from many causes, including infections or "
        "allergies. Staying hydrated and resting can help. If the cough "
        "is persistent, severe, or accompanied by breathing difficulty, "
        "please see a doctor."
    ),
    "headache": (
        "Headaches can often be managed with rest, hydration, and a "
        "quiet environment. Frequent or severe headaches should be "
        "evaluated by a doctor."
    ),
    "bmi": (
        "BMI (Body Mass Index) is a general screening measure calculated "
        "from your height and weight. It is used to broadly categorize "
        "weight status but does not account for muscle mass or body "
        "composition."
    ),
    "diet": (
        "A balanced diet generally includes a variety of fruits, "
        "vegetables, whole grains, lean protein, and healthy fats, while "
        "limiting processed foods, added sugar, and excess salt."
    ),
    "exercise": (
        "Regular physical activity, such as at least 150 minutes of "
        "moderate exercise per week, supports overall health, weight "
        "management, and cardiovascular fitness."
    ),
    "sleep": (
        "Most adults benefit from 7-9 hours of quality sleep each night. "
        "Maintaining a consistent sleep schedule can improve overall "
        "health and well-being."
    ),
    "hydration": (
        "Staying well hydrated, generally around 8 glasses of water a "
        "day for most adults, supports digestion, circulation, and "
        "overall bodily function."
    ),
    "lifestyle": (
        "Healthy lifestyle habits include regular exercise, a balanced "
        "diet, adequate sleep, stress management, and avoiding smoking "
        "or excessive alcohol consumption."
    ),
    "medicine": (
        "I can share general educational information about medicine "
        "categories, but I cannot prescribe medications or recommend "
        "dosages. Always follow the guidance of a licensed healthcare "
        "provider or pharmacist regarding any medication."
    ),
    "doctor": (
        "If you have ongoing symptoms, concerns about a diagnosis, or "
        "questions about treatment, it's best to consult a qualified "
        "doctor or specialist who can evaluate your specific situation."
    ),
}

_GREETING_KEYWORDS: Final[tuple[str, ...]] = ("hello", "hi", "hey", "good morning", "good evening")
_THANKS_KEYWORDS: Final[tuple[str, ...]] = ("thank", "thanks")


def _validate_user_message(user_message: str) -> str:
    """Validate the user's chat message.

    Args:
        user_message: The message text provided by the user.

    Returns:
        The stripped, validated message.

    Raises:
        HealthChatbotError: If the message is empty or not a string.
    """
    if not isinstance(user_message, str) or not user_message.strip():
        message = "User message cannot be empty."
        logger.error(message)
        raise HealthChatbotError(message)

    return user_message.strip()


def _validate_predicted_disease(predicted_disease: str | None) -> str | None:
    """Validate the optional predicted disease value.

    Args:
        predicted_disease: The predicted disease name, or None.

    Returns:
        The stripped predicted disease, or None if not provided.

    Raises:
        HealthChatbotError: If provided but not a string.
    """
    if predicted_disease is None:
        return None

    if not isinstance(predicted_disease, str):
        message = "Predicted disease must be a string when provided."
        logger.error(message)
        raise HealthChatbotError(message)

    return predicted_disease.strip() or None


def _validate_bmi(bmi: float | None) -> float | None:
    """Validate the optional BMI value.

    Args:
        bmi: The BMI value, or None.

    Returns:
        The validated BMI value, or None if not provided.

    Raises:
        HealthChatbotError: If provided but not a positive numeric
            value.
    """
    if bmi is None:
        return None

    if isinstance(bmi, bool) or not isinstance(bmi, (int, float)):
        message = "BMI must be a numeric value when provided."
        logger.error(message)
        raise HealthChatbotError(message)

    if bmi <= 0:
        message = "BMI must be a positive numeric value."
        logger.error(message)
        raise HealthChatbotError(message)

    return float(bmi)


def _contains_emergency_keywords(message: str) -> bool:
    """Check whether a message mentions emergency symptoms.

    Args:
        message: The lowercase message text to check.

    Returns:
        True if an emergency keyword is present, False otherwise.
    """
    return any(keyword in message for keyword in _EMERGENCY_KEYWORDS)


def _detect_topic(message: str) -> str | None:
    """Detect the most relevant known topic within a user message.

    Args:
        message: The lowercase message text to inspect.

    Returns:
        The matched topic key, or None if no topic matched.
    """
    for topic, keywords in _TOPIC_KEYWORDS.items():
        if any(keyword in message for keyword in keywords):
            return topic
    return None


def _build_rule_based_response(
    user_message: str,
    predicted_disease: str | None,
    bmi: float | None,
) -> str:
    """Generate a rule-based chatbot response.

    Args:
        user_message: The validated user message.
        predicted_disease: The validated predicted disease, if any.
        bmi: The validated BMI value, if any.

    Returns:
        A user-friendly, safe response string.
    """
    normalized_message = user_message.lower()

    if _contains_emergency_keywords(normalized_message):
        logger.info("Emergency keywords detected in user message.")
        return (
            f"{EMERGENCY_DISCLAIMER}\n\n{MEDICAL_DISCLAIMER}"
        )

    if any(keyword in normalized_message for keyword in _GREETING_KEYWORDS):
        return (
            "Hello! I'm your AI Health Assistant. You can ask me about "
            "symptoms, BMI, diet, exercise, sleep, hydration, lifestyle, "
            "or general medicine information. How can I help you today?"
        )

    if any(keyword in normalized_message for keyword in _THANKS_KEYWORDS):
        return "You're welcome! Let me know if you have any other health questions."

    topic = _detect_topic(normalized_message)

    response_parts: list[str] = []

    if topic == "bmi" and bmi is not None:
        response_parts.append(
            f"Your current BMI is {bmi:.2f}. {_TOPIC_RESPONSES['bmi']}"
        )
    elif topic is not None:
        response_parts.append(_TOPIC_RESPONSES[topic])
    else:
        response_parts.append(
            "I can share general educational information about symptoms, "
            "BMI, diet, exercise, sleep, hydration, lifestyle habits, and "
            "medicine categories. Could you tell me more about what "
            "you'd like to know?"
        )

    if predicted_disease:
        response_parts.append(
            f"Based on your recent prediction of '{predicted_disease}', "
            "it's a good idea to discuss this with a doctor for proper "
            "evaluation and care."
        )

    needs_disclaimer_topics = {"medicine", "diabetes", "hypertension", "doctor"}
    if topic in needs_disclaimer_topics or predicted_disease:
        response_parts.append(MEDICAL_DISCLAIMER)

    return "\n\n".join(response_parts)


def _get_ai_api_credentials() -> tuple[str | None, str | None]:
    """Read AI provider credentials from environment variables.

    Returns:
        A tuple of (provider, api_key). Either or both may be None if
        not configured. Values are never logged.
    """
    provider = os.environ.get(AI_API_PROVIDER_ENV_VAR)
    api_key = os.environ.get(AI_API_KEY_ENV_VAR)
    return provider, api_key


def _try_ai_api_response(
    user_message: str,
    predicted_disease: str | None,
    bmi: float | None,
) -> str | None:
    """Attempt to generate a response using an external AI provider.

    This is a safe placeholder for optional AI API integration (e.g.,
    OpenAI or Gemini). It reads credentials from environment variables
    and never hard-codes secrets. If credentials are missing or any
    error occurs, it returns None so the caller can fall back to the
    rule-based system.

    Args:
        user_message: The validated user message.
        predicted_disease: The validated predicted disease, if any.
        bmi: The validated BMI value, if any.

    Returns:
        The AI-generated response string, or None if unavailable.
    """
    provider, api_key = _get_ai_api_credentials()

    if not provider or not api_key:
        logger.info("No AI API credentials configured. Using rule-based fallback.")
        return None

    try:
        # NOTE: Actual API integration is intentionally not implemented
        # here to avoid introducing hard dependencies or network calls
        # by default. This block is a safe extension point for future
        # integration with a provider such as OpenAI or Gemini.
        logger.info("AI API provider '%s' configured, but integration is not implemented.", provider)
        return None
    except Exception:  # noqa: BLE001
        logger.exception("AI API call failed. Falling back to rule-based response.")
        return None


def get_chatbot_response(
    user_message: str,
    predicted_disease: str | None = None,
    bmi: float | None = None,
) -> str:
    """Generate a chatbot response to a user's health-related question.

    This function first attempts to use an external AI provider if
    credentials are configured via environment variables. If no
    credentials are available, or the AI call fails for any reason, it
    falls back to a safe, rule-based response system.

    Args:
        user_message: The user's chat message. Must be a non-empty
            string.
        predicted_disease: The most recently predicted disease, if
            available, used to personalize the response.
        bmi: The user's current BMI value, if available, used to
            personalize BMI-related responses. Must be positive if
            provided.

    Returns:
        A clear, user-friendly response string, including a medical
        disclaimer where appropriate.

    Raises:
        HealthChatbotError: If ``user_message`` is empty or not a
            string, ``predicted_disease`` is provided but not a string,
            or ``bmi`` is provided but not a positive numeric value.
    """
    validated_message = _validate_user_message(user_message)
    validated_disease = _validate_predicted_disease(predicted_disease)
    validated_bmi = _validate_bmi(bmi)

    logger.info("User question received (length=%d characters).", len(validated_message))

    ai_response = _try_ai_api_response(validated_message, validated_disease, validated_bmi)

    if ai_response:
        logger.info("Response generated using AI API provider.")
        return ai_response

    response = _build_rule_based_response(validated_message, validated_disease, validated_bmi)
    logger.info("Response generated using rule-based system.")

    return response


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        disease_question = get_chatbot_response(
            user_message="What can you tell me about diabetes?",
            predicted_disease="Diabetes",
        )
        logger.info("Disease question response:\n%s", disease_question)

        bmi_question = get_chatbot_response(
            user_message="What does my BMI mean?",
            bmi=27.4,
        )
        logger.info("BMI question response:\n%s", bmi_question)

        lifestyle_question = get_chatbot_response(
            user_message="Any tips for a healthier lifestyle?",
        )
        logger.info("Lifestyle question response:\n%s", lifestyle_question)

        medicine_question = get_chatbot_response(
            user_message="Can you tell me what medicine to take?",
        )
        logger.info("Medicine question response:\n%s", medicine_question)
    except HealthChatbotError:
        logger.exception("Chatbot response generation failed.")
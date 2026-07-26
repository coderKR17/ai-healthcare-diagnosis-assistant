"""Prediction module for the AI-Powered Healthcare Diagnosis Assistant.

This module loads a previously trained RandomForestClassifier model and
uses it to predict a disease based on a dictionary of symptoms provided by
the caller. It is completely framework independent and can be reused
across different interfaces (CLI, API, web frameworks, etc.).

Typical usage example:
    from src.predictor import predict_disease

    symptoms = {"itching": 1, "skin_rash": 1, "headache": 0}
    disease = predict_disease(symptoms)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_MODEL_PATH: Final[Path] = Path("models/disease_model.pkl")
DEFAULT_DATA_PATH: Final[Path] = Path("data/Training.csv")
TARGET_COLUMN: Final[str] = "prognosis"


class PredictionError(Exception):
    """Raised when the disease prediction pipeline fails.

    This exception wraps any underlying error encountered during model
    loading, feature loading, input validation, or prediction, providing a
    single, consistent error type for callers to handle.
    """


def _load_model(model_path: Path) -> RandomForestClassifier:
    """Load a trained model from disk.

    Args:
        model_path: Path to the serialized model file.

    Returns:
        The deserialized RandomForestClassifier instance.

    Raises:
        PredictionError: If the model file does not exist or cannot be
            loaded.
    """
    if not model_path.exists():
        message = f"Model file not found at: {model_path}"
        logger.error(message)
        raise PredictionError(message)

    if not model_path.is_file():
        message = f"Model path is not a file: {model_path}"
        logger.error(message)
        raise PredictionError(message)

    try:
        with model_path.open("rb") as file_handle:
            model = pickle.load(file_handle)
    except (OSError, pickle.PickleError) as exc:
        message = f"Failed to load model from: {model_path}"
        logger.error(message)
        raise PredictionError(message) from exc

    logger.info("Model loaded successfully from %s", model_path)
    return model


def _load_feature_names(data_path: Path) -> list[str]:
    """Load the list of feature names used to train the model.

    Args:
        data_path: Path to the training dataset CSV file.

    Returns:
        A list of feature column names, excluding the target column.

    Raises:
        PredictionError: If the dataset file does not exist, is empty, or
            cannot be read.
    """
    if not data_path.exists():
        message = f"Training data file not found at: {data_path}"
        logger.error(message)
        raise PredictionError(message)

    try:
        dataframe = pd.read_csv(data_path)
    except pd.errors.EmptyDataError as exc:
        message = f"Training data file contains no data: {data_path}"
        logger.error(message)
        raise PredictionError(message) from exc
    except (pd.errors.ParserError, OSError, ValueError) as exc:
        message = f"Failed to read training data file: {data_path}"
        logger.error(message)
        raise PredictionError(message) from exc

    if dataframe.empty:
        message = f"Training data loaded from {data_path} is empty."
        logger.error(message)
        raise PredictionError(message)

    feature_names = [
        column for column in dataframe.columns if column != TARGET_COLUMN
    ]

    if not feature_names:
        message = "No feature columns found in training data."
        logger.error(message)
        raise PredictionError(message)

    logger.info("Loaded %d feature names from %s", len(feature_names), data_path)
    return feature_names


def _build_feature_vector(
    symptoms: dict[str, int],
    feature_names: list[str],
) -> pd.DataFrame:
    """Build a single-row feature vector from the provided symptoms.

    Missing symptoms are automatically filled with 0. Any symptom keys not
    recognized as model features are ignored.

    Args:
        symptoms: Dictionary mapping symptom names to 0 or 1.
        feature_names: The complete list of feature names expected by the
            model, in the correct order.

    Returns:
        A single-row DataFrame with columns ordered to match
        ``feature_names``.

    Raises:
        PredictionError: If the feature vector cannot be constructed.
    """
    try:
        feature_values = {
            feature: int(symptoms.get(feature, 0)) for feature in feature_names
        }
        feature_vector = pd.DataFrame([feature_values], columns=feature_names)
    except (TypeError, ValueError) as exc:
        message = "Failed to build feature vector from provided symptoms."
        logger.error(message)
        raise PredictionError(message) from exc

    return feature_vector


def predict_disease(
    symptoms: dict[str, int],
    model_path: Path | str = DEFAULT_MODEL_PATH,
    data_path: Path | str = DEFAULT_DATA_PATH,
) -> str:
    """Predict a disease based on a dictionary of symptoms.

    This function orchestrates the full prediction pipeline: loading the
    trained model, loading the expected feature names, validating and
    filling the input symptoms, and running the prediction.

    Args:
        symptoms: Dictionary mapping symptom names to 0 or 1, where 1
            indicates the symptom is present and 0 indicates it is absent.
            Any symptoms required by the model but missing from this
            dictionary are automatically filled with 0.
        model_path: Path to the serialized trained model file. Defaults to
            ``models/disease_model.pkl``.
        data_path: Path to the training dataset CSV file, used to
            determine the expected feature names. Defaults to
            ``data/Training.csv``.

    Returns:
        The predicted disease name as a string.

    Raises:
        PredictionError: If any step of the prediction pipeline fails.
    """
    if not isinstance(symptoms, dict):
        message = "Symptoms input must be a dictionary."
        logger.error(message)
        raise PredictionError(message)

    model_path = Path(model_path)
    data_path = Path(data_path)

    logger.info("Starting disease prediction pipeline.")

    model = _load_model(model_path)
    feature_names = _load_feature_names(data_path)
    feature_vector = _build_feature_vector(symptoms, feature_names)

    try:
        prediction = model.predict(feature_vector)
    except Exception as exc:
        message = "Failed to generate prediction from the trained model."
        logger.error(message)
        raise PredictionError(message) from exc

    if len(prediction) == 0:
        message = "Model returned an empty prediction result."
        logger.error(message)
        raise PredictionError(message)

    predicted_disease = str(prediction[0])
    logger.info("Prediction completed successfully: %s", predicted_disease)

    return predicted_disease


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_symptoms: dict[str, int] = {
        "itching": 1,
        "skin_rash": 1,
        "headache": 0,
    }
    try:
        result = predict_disease(sample_symptoms)
        logger.info("Predicted disease: %s", result)
    except PredictionError:
        logger.exception("Prediction failed.")
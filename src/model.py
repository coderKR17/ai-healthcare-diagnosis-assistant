"""Model training module for the AI-Powered Healthcare Diagnosis Assistant.

This module is responsible for loading the training dataset, validating its
structure, splitting it into features and target, training a
RandomForestClassifier, evaluating its accuracy, and persisting the trained
model to disk.

The module is completely framework independent and can be reused across
different interfaces (CLI, API, web frameworks, etc.).

Typical usage example:
    from src.model import train_model

    accuracy = train_model()
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_DATA_PATH: Final[Path] = Path("data/Training.csv")
DEFAULT_MODEL_PATH: Final[Path] = Path("models/disease_model.pkl")
TARGET_COLUMN: Final[str] = "prognosis"
DEFAULT_TEST_SIZE: Final[float] = 0.2
DEFAULT_RANDOM_STATE: Final[int] = 42


class ModelTrainingError(Exception):
    """Raised when the model training pipeline fails.

    This exception wraps any underlying error encountered during dataset
    loading, validation, splitting, training, evaluation, or persistence,
    providing a single, consistent error type for callers to handle.
    """


def _validate_dataset(data_path: Path) -> None:
    """Validate the existence and integrity of the dataset file.

    Args:
        data_path: Path to the dataset CSV file.

    Raises:
        ModelTrainingError: If the file does not exist, is empty, or does
            not contain the required target column.
    """
    if not data_path.exists():
        message = f"Dataset file not found at: {data_path}"
        logger.error(message)
        raise ModelTrainingError(message)

    if not data_path.is_file():
        message = f"Dataset path is not a file: {data_path}"
        logger.error(message)
        raise ModelTrainingError(message)

    if data_path.stat().st_size == 0:
        message = f"Dataset file is empty: {data_path}"
        logger.error(message)
        raise ModelTrainingError(message)


def _load_dataset(data_path: Path) -> pd.DataFrame:
    """Load the dataset from a CSV file into a DataFrame.

    Args:
        data_path: Path to the dataset CSV file.

    Returns:
        The loaded dataset as a pandas DataFrame.

    Raises:
        ModelTrainingError: If the dataset cannot be read or is empty
            after loading, or if the required target column is missing.
    """
    try:
        dataframe = pd.read_csv(data_path)
    except pd.errors.EmptyDataError as exc:
        message = f"Dataset file contains no data: {data_path}"
        logger.error(message)
        raise ModelTrainingError(message) from exc
    except (pd.errors.ParserError, OSError, ValueError) as exc:
        message = f"Failed to read dataset file: {data_path}"
        logger.error(message)
        raise ModelTrainingError(message) from exc

    if dataframe.empty:
        message = f"Dataset loaded from {data_path} is empty."
        logger.error(message)
        raise ModelTrainingError(message)

    if TARGET_COLUMN not in dataframe.columns:
        message = (
            f"Required target column '{TARGET_COLUMN}' not found in "
            f"dataset columns: {list(dataframe.columns)}"
        )
        logger.error(message)
        raise ModelTrainingError(message)

    logger.info(
        "Dataset loaded successfully from %s with shape %s",
        data_path,
        dataframe.shape,
    )
    return dataframe


def _split_features_and_target(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataset into features and target.

    Args:
        dataframe: The full dataset containing features and target.

    Returns:
        A tuple of (features, target).

    Raises:
        ModelTrainingError: If the split operation fails.
    """
    try:
        features = dataframe.drop(columns=[TARGET_COLUMN])
        target = dataframe[TARGET_COLUMN]
    except KeyError as exc:
        message = f"Failed to split features and target: {exc}"
        logger.error(message)
        raise ModelTrainingError(message) from exc

    return features, target


def _train_random_forest(
    features_train: pd.DataFrame,
    target_train: pd.Series,
    random_state: int,
) -> RandomForestClassifier:
    """Train a RandomForestClassifier on the provided training data.

    Args:
        features_train: Training feature set.
        target_train: Training target labels.
        random_state: Random state for reproducibility.

    Returns:
        The trained RandomForestClassifier instance.

    Raises:
        ModelTrainingError: If training fails.
    """
    try:
        model = RandomForestClassifier(random_state=random_state)
        model.fit(features_train, target_train)
    except Exception as exc:
        message = "Failed to train RandomForestClassifier."
        logger.error(message)
        raise ModelTrainingError(message) from exc

    return model


def _save_model(model: RandomForestClassifier, model_path: Path) -> None:
    """Persist the trained model to disk using pickle.

    Args:
        model: The trained model instance to persist.
        model_path: Destination path for the serialized model.

    Raises:
        ModelTrainingError: If the model cannot be saved.
    """
    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as file_handle:
            pickle.dump(model, file_handle)
    except (OSError, pickle.PickleError) as exc:
        message = f"Failed to save trained model to: {model_path}"
        logger.error(message)
        raise ModelTrainingError(message) from exc

    logger.info("Trained model saved successfully to %s", model_path)


def train_model(
    data_path: Path | str = DEFAULT_DATA_PATH,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> float:
    """Train a RandomForestClassifier on the healthcare diagnosis dataset.

    This function orchestrates the full training pipeline: loading and
    validating the dataset, splitting it into features and target,
    performing a train/test split, training the model, evaluating its
    accuracy, and persisting the trained model to disk.

    Args:
        data_path: Path to the training dataset CSV file. Defaults to
            ``data/Training.csv``.
        model_path: Destination path for the saved model file. Defaults to
            ``models/disease_model.pkl``.
        test_size: Proportion of the dataset to include in the test split.
            Defaults to 0.2.
        random_state: Random state for reproducibility. Defaults to 42.

    Returns:
        The accuracy score of the trained model on the test split.

    Raises:
        ModelTrainingError: If any step of the training pipeline fails.
    """
    data_path = Path(data_path)
    model_path = Path(model_path)

    logger.info("Starting model training pipeline.")

    _validate_dataset(data_path)
    dataframe = _load_dataset(data_path)
    features, target = _split_features_and_target(dataframe)

    try:
        features_train, features_test, target_train, target_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
        )
    except ValueError as exc:
        message = "Failed to split dataset into train and test sets."
        logger.error(message)
        raise ModelTrainingError(message) from exc

    model = _train_random_forest(features_train, target_train, random_state)

    try:
        predictions = model.predict(features_test)
        accuracy = accuracy_score(target_test, predictions)
    except Exception as exc:
        message = "Failed to evaluate model accuracy."
        logger.error(message)
        raise ModelTrainingError(message) from exc

    logger.info("Model trained successfully with accuracy: %.4f", accuracy)

    _save_model(model, model_path)

    return float(accuracy)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        result_accuracy = train_model()
        logger.info("Training completed. Final accuracy: %.4f", result_accuracy)
    except ModelTrainingError:
        logger.exception("Model training failed.")
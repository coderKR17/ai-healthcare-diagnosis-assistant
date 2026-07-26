"""Patient data module for the AI-Powered Healthcare Diagnosis Assistant.

This module defines the ``Patient`` dataclass, which stores and validates
patient information, and provides utility methods for calculating BMI and
determining BMI category. It is completely framework independent and can
be reused across different interfaces (CLI, API, web frameworks, etc.).

Typical usage example:
    from src.patient import Patient

    patient = Patient(
        full_name="John Doe",
        age=30,
        gender="Male",
        height_cm=175.0,
        weight_kg=70.0,
        blood_group="O+",
        allergies="None",
        existing_diseases="None",
        smoking=False,
        alcohol=False,
        phone_number="9876543210",
        email="john.doe@example.com",
    )
    bmi = patient.calculate_bmi()
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Final

logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

VALID_GENDERS: Final[tuple[str, ...]] = ("Male", "Female", "Other")
MIN_AGE: Final[int] = 1
MAX_AGE: Final[int] = 120
MIN_HEIGHT_CM: Final[float] = 50.0
MAX_HEIGHT_CM: Final[float] = 250.0
MIN_WEIGHT_KG: Final[float] = 2.0
MAX_WEIGHT_KG: Final[float] = 300.0
MIN_PHONE_LENGTH: Final[int] = 7
MAX_PHONE_LENGTH: Final[int] = 15
EMAIL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
)


class PatientValidationError(Exception):
    """Raised when patient data fails validation.

    This exception is raised whenever a ``Patient`` instance is created or
    modified with data that does not satisfy the required validation
    rules (e.g., invalid age, height, weight, email, phone number, or
    gender).
    """


@dataclass
class Patient:
    """Represents a patient and their medical profile.

    Attributes:
        full_name: The patient's full name. Cannot be empty.
        age: The patient's age in years. Must be between 1 and 120.
        gender: The patient's gender. Must be one of 'Male', 'Female',
            or 'Other'.
        height_cm: The patient's height in centimeters. Must be between
            50 and 250.
        weight_kg: The patient's weight in kilograms. Must be between 2
            and 300.
        blood_group: The patient's blood group (e.g., 'O+', 'AB-').
        allergies: A description of the patient's known allergies.
        existing_diseases: A description of the patient's existing
            diseases or conditions.
        smoking: Whether the patient smokes.
        alcohol: Whether the patient consumes alcohol.
        phone_number: The patient's contact phone number.
        email: The patient's email address.
    """

    full_name: str
    age: int
    gender: str
    height_cm: float
    weight_kg: float
    blood_group: str
    allergies: str
    existing_diseases: str
    smoking: bool
    alcohol: bool
    phone_number: str
    email: str
    _validated: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate patient data immediately after initialization.

        Raises:
            PatientValidationError: If any field fails validation.
        """
        logger.info("Validating patient data for '%s'.", self.full_name)
        self._validate_name()
        self._validate_age()
        self._validate_gender()
        self._validate_height()
        self._validate_weight()
        self._validate_email()
        self._validate_phone_number()
        self._validated = True
        logger.info("Patient data validated successfully for '%s'.", self.full_name)

    def _validate_name(self) -> None:
        """Validate that the patient's name is not empty.

        Raises:
            PatientValidationError: If the name is empty or whitespace only.
        """
        if not self.full_name or not self.full_name.strip():
            message = "Patient name cannot be empty."
            logger.error(message)
            raise PatientValidationError(message)

    def _validate_age(self) -> None:
        """Validate that the patient's age is within the allowed range.

        Raises:
            PatientValidationError: If the age is outside [1, 120].
        """
        if not isinstance(self.age, int) or isinstance(self.age, bool):
            message = "Patient age must be an integer."
            logger.error(message)
            raise PatientValidationError(message)

        if not (MIN_AGE <= self.age <= MAX_AGE):
            message = (
                f"Patient age must be between {MIN_AGE} and {MAX_AGE}, "
                f"got {self.age}."
            )
            logger.error(message)
            raise PatientValidationError(message)

    def _validate_gender(self) -> None:
        """Validate that the patient's gender is one of the allowed values.

        Raises:
            PatientValidationError: If gender is not 'Male', 'Female', or
                'Other'.
        """
        if self.gender not in VALID_GENDERS:
            message = (
                f"Patient gender must be one of {VALID_GENDERS}, "
                f"got '{self.gender}'."
            )
            logger.error(message)
            raise PatientValidationError(message)

    def _validate_height(self) -> None:
        """Validate that the patient's height is within the allowed range.

        Raises:
            PatientValidationError: If height is outside [50, 250] cm.
        """
        if not (MIN_HEIGHT_CM <= self.height_cm <= MAX_HEIGHT_CM):
            message = (
                f"Patient height must be between {MIN_HEIGHT_CM} and "
                f"{MAX_HEIGHT_CM} cm, got {self.height_cm}."
            )
            logger.error(message)
            raise PatientValidationError(message)

    def _validate_weight(self) -> None:
        """Validate that the patient's weight is within the allowed range.

        Raises:
            PatientValidationError: If weight is outside [2, 300] kg.
        """
        if not (MIN_WEIGHT_KG <= self.weight_kg <= MAX_WEIGHT_KG):
            message = (
                f"Patient weight must be between {MIN_WEIGHT_KG} and "
                f"{MAX_WEIGHT_KG} kg, got {self.weight_kg}."
            )
            logger.error(message)
            raise PatientValidationError(message)

    def _validate_email(self) -> None:
        """Validate that the patient's email address is well formed.

        Raises:
            PatientValidationError: If the email format is invalid.
        """
        if not self.email or not EMAIL_PATTERN.match(self.email):
            message = f"Invalid email address format: '{self.email}'."
            logger.error(message)
            raise PatientValidationError(message)

    def _validate_phone_number(self) -> None:
        """Validate that the patient's phone number has a valid length.

        Only digit characters are counted; separators such as spaces,
        hyphens, or a leading '+' are stripped before validation.

        Raises:
            PatientValidationError: If the phone number length is outside
                the allowed range.
        """
        digits_only = re.sub(r"\D", "", self.phone_number or "")

        if not (MIN_PHONE_LENGTH <= len(digits_only) <= MAX_PHONE_LENGTH):
            message = (
                f"Patient phone number must contain between "
                f"{MIN_PHONE_LENGTH} and {MAX_PHONE_LENGTH} digits, "
                f"got {len(digits_only)}."
            )
            logger.error(message)
            raise PatientValidationError(message)

    def calculate_bmi(self) -> float:
        """Calculate the patient's Body Mass Index (BMI).

        Returns:
            The BMI value rounded to 2 decimal places.

        Raises:
            PatientValidationError: If BMI cannot be calculated due to
                invalid height or weight values.
        """
        try:
            height_m = self.height_cm / 100
            bmi = self.weight_kg / (height_m ** 2)
        except (ZeroDivisionError, ArithmeticError) as exc:
            message = "Failed to calculate BMI due to invalid measurements."
            logger.error(message)
            raise PatientValidationError(message) from exc

        rounded_bmi = round(bmi, 2)
        logger.info("Calculated BMI for '%s': %.2f", self.full_name, rounded_bmi)
        return rounded_bmi

    def bmi_category(self) -> str:
        """Determine the patient's BMI category.

        Returns:
            One of 'Underweight', 'Normal', 'Overweight', or 'Obese'.
        """
        bmi = self.calculate_bmi()

        if bmi < 18.5:
            category = "Underweight"
        elif bmi < 25:
            category = "Normal"
        elif bmi < 30:
            category = "Overweight"
        else:
            category = "Obese"

        logger.info("BMI category for '%s': %s", self.full_name, category)
        return category

    def to_dict(self) -> dict[str, Any]:
        """Convert the patient's information into a dictionary.

        Returns:
            A dictionary containing all patient fields along with the
            calculated BMI and BMI category.
        """
        return {
            "full_name": self.full_name,
            "age": self.age,
            "gender": self.gender,
            "height_cm": self.height_cm,
            "weight_kg": self.weight_kg,
            "blood_group": self.blood_group,
            "allergies": self.allergies,
            "existing_diseases": self.existing_diseases,
            "smoking": self.smoking,
            "alcohol": self.alcohol,
            "phone_number": self.phone_number,
            "email": self.email,
            "bmi": self.calculate_bmi(),
            "bmi_category": self.bmi_category(),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        sample_patient = Patient(
            full_name="John Doe",
            age=30,
            gender="Male",
            height_cm=175.0,
            weight_kg=70.0,
            blood_group="O+",
            allergies="None",
            existing_diseases="None",
            smoking=False,
            alcohol=False,
            phone_number="9876543210",
            email="john.doe@example.com",
        )
        logger.info("Patient BMI: %.2f", sample_patient.calculate_bmi())
        logger.info("Patient BMI Category: %s", sample_patient.bmi_category())
        logger.info("Patient Dictionary: %s", sample_patient.to_dict())
    except PatientValidationError:
        logger.exception("Patient validation failed.")
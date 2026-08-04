"""Streamlit application for the AI-Powered Healthcare Diagnosis Assistant.

This application provides a professional web interface that allows users to:
    1. Train a RandomForestClassifier disease prediction model.
    2. Select symptoms and predict the most likely disease.

The application relies on the following independent, reusable modules:
    - src.model: Handles dataset loading, validation, and model training.
    - src.predictor: Handles trained model loading and disease prediction.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from src.model import ModelTrainingError, train_model
from src.predictor import PredictionError, predict_disease
from src.patient import Patient, PatientValidationError
from src.bmi import (
    calculate_bmi,
    get_bmi_category,
    get_health_risk,
    get_health_tip,
    BMICalculationError,
)
from src.medicine import (
    get_medicine_recommendation,
    MedicineRecommendationError,
)
from src.lab_tests import (
    get_lab_test_recommendation,
    LabTestRecommendationError,
)
from src.history import (
    create_history_record,
    append_history,
    clear_history,
    HistoryError,
)

logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

TRAINING_DATA_PATH: str = "data/Training.csv"
TARGET_COLUMN: str = "prognosis"


def configure_page() -> None:
    """Configure Streamlit page settings such as title and layout."""
    st.set_page_config(
        page_title="AI-Powered Healthcare Diagnosis Assistant",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_header() -> None:
    """Render the project title and description on the main page."""
    st.title("🩺 AI-Powered Healthcare Diagnosis Assistant")
    st.markdown(
        """
        Welcome to the **AI-Powered Healthcare Diagnosis Assistant** — a
        machine learning based tool that predicts possible diseases based
        on the symptoms you provide.

        Use the sections below to:
        - **Train** the disease prediction model on the latest dataset.
        - **Predict** a disease by selecting your symptoms.

        > ⚠️ **Disclaimer:** This tool is for educational and informational
        > purposes only and is **not** a substitute for professional
        > medical advice, diagnosis, or treatment.
        """
    )
    st.divider()


def render_sidebar() -> None:
    """Render the sidebar with project information and instructions."""
    with st.sidebar:
        st.header("About")
        st.markdown(
            """
            **AI-Powered Healthcare Diagnosis Assistant** uses a
            RandomForestClassifier trained on symptom-disease data to
            predict likely diseases from user-selected symptoms.
            """
        )

        st.header("How to Use")
        st.markdown(
            """
            1. Click **Train Disease Prediction Model** to train the model
               (only required once, or after updating the dataset).
            2. Select your symptoms in the **Disease Prediction** section.
            3. Click **Predict Disease** to view the result.
            """
        )

        st.header("Project Info")
        st.markdown(
            """
            - **Model:** RandomForestClassifier
            - **Dataset:** `data/Training.csv`
            - **Saved Model:** `models/disease_model.pkl`
            """
        )


def render_training_section() -> None:
    """Render the model training section and handle training actions."""
    st.header("🧠 Train Disease Prediction Model")
    st.write("Train the RandomForest model on the latest training dataset.")

    if st.button("Train Disease Prediction Model"):
        with st.spinner("Training model, please wait..."):
            try:
                accuracy = train_model()
                st.success("✅ Model trained successfully!")
                st.metric(
                    label="Training Accuracy",
                    value=f"{accuracy * 100:.2f}%",
                )
                logger.info("Model trained with accuracy: %.4f", accuracy)
            except ModelTrainingError as error:
                st.error(f"Model training failed: {error}")
                logger.error("Model training failed: %s", error)
            except Exception as error:  # noqa: BLE001
                st.error(f"An unexpected error occurred during training: {error}")
                logger.exception("Unexpected error during model training.")

    st.divider()

def render_patient_information_section() -> None:
    """Render the patient information form and handle saving actions."""
    st.header("🧾 Patient Information")
    st.write("Please provide the patient's details below.")

    with st.form(key="patient_information_form"):
        col1, col2 = st.columns(2)

        with col1:
            full_name = st.text_input("Full Name")
            age = st.number_input(
                "Age", min_value=1, max_value=120, value=25, step=1
            )
            gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
            height_cm = st.number_input(
                "Height (cm)", min_value=50.0, max_value=250.0, value=170.0
            )
            weight_kg = st.number_input(
                "Weight (kg)", min_value=2.0, max_value=300.0, value=65.0
            )
            blood_group = st.selectbox(
                "Blood Group",
                options=["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
            )

        with col2:
            allergies = st.text_area("Allergies", placeholder="e.g., Pollen, Dust")
            existing_diseases = st.text_area(
                "Existing Diseases", placeholder="e.g., Diabetes, Hypertension"
            )
            smoking = st.checkbox("Smoking")
            alcohol = st.checkbox("Alcohol")
            phone_number = st.text_input("Phone Number")
            email = st.text_input("Email")

        submitted = st.form_submit_button("Save Patient Information")

    if submitted:
        try:
            patient = Patient(
                full_name=full_name,
                age=int(age),
                gender=gender,
                height_cm=float(height_cm),
                weight_kg=float(weight_kg),
                blood_group=blood_group,
                allergies=allergies,
                existing_diseases=existing_diseases,
                smoking=smoking,
                alcohol=alcohol,
                phone_number=phone_number,
                email=email,
            )

            st.session_state["patient"] = patient

            st.success("✅ Patient information saved successfully!")
            st.metric(label="BMI", value=f"{patient.calculate_bmi():.2f}")
            st.metric(label="BMI Category", value=patient.bmi_category())
            logger.info("Patient information saved for '%s'.", patient.full_name)

        except PatientValidationError as error:
            st.error(f"Patient validation failed: {error}")
            logger.error("Patient validation failed: %s", error)
        except Exception as error:  # noqa: BLE001
            st.error(f"An unexpected error occurred while saving patient data: {error}")
            logger.exception("Unexpected error while saving patient information.")

    st.divider()

def render_bmi_section() -> None:
    """Render the BMI and health risk assessment section.

    Reads the previously saved patient data from ``st.session_state``,
    calculates BMI, and displays the BMI value, category, associated
    health risk, and a professional health tip.
    """
    st.header("📏 BMI & Health Risk Assessment")

    patient = st.session_state.get("patient")

    if patient is None:
        st.warning(
            "⚠️ No patient information found. Please fill out and save "
            "the **Patient Information** section first."
        )
        st.divider()
        return

    try:
        bmi = calculate_bmi(
            height_cm=patient.height_cm, weight_kg=patient.weight_kg
        )
        category = get_bmi_category(bmi)
        risk = get_health_risk(bmi)
        tip = get_health_tip(category)

        st.success(f"✅ BMI assessment completed for **{patient.full_name}**.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="BMI", value=f"{bmi:.2f}")
        with col2:
            st.metric(label="BMI Category", value=category)
        with col3:
            st.metric(label="Health Risk", value=risk)

        st.info(f"💡 **Health Tip:** {tip}")

        logger.info(
            "BMI assessment displayed for '%s': BMI=%.2f, category=%s, "
            "risk=%s",
            patient.full_name,
            bmi,
            category,
            risk,
        )

    except BMICalculationError as error:
        st.error(f"BMI calculation failed: {error}")
        logger.error("BMI calculation failed: %s", error)
    except Exception as error:  # noqa: BLE001
        st.error(f"An unexpected error occurred during BMI assessment: {error}")
        logger.exception("Unexpected error during BMI assessment.")

    st.divider()

def render_medicine_recommendation_section() -> None:
    """Render the medicine recommendation section.

    Reads the previously predicted disease from ``st.session_state`` and
    displays commonly associated medicine categories and precautions for
    educational purposes only.
    """
    st.header("💊 Medicine Recommendation")

    predicted_disease = st.session_state.get("predicted_disease")

    if not predicted_disease:
        st.warning(
            "⚠️ No predicted disease found. Please predict a disease in "
            "the **Disease Prediction** section first."
        )
        st.divider()
        return

    try:
        recommendation = get_medicine_recommendation(predicted_disease)

        st.success(
            f"✅ Medicine recommendation retrieved for "
            f"**{recommendation['disease']}**."
        )

        st.subheader("Disease")
        st.write(recommendation["disease"])

        st.subheader("Common Medicines")
        medicines_markdown = "\n".join(
            f"- {medicine}" for medicine in recommendation["common_medicines"]
        )
        st.markdown(medicines_markdown)

        st.subheader("Precautions")
        for precaution in recommendation["precautions"]:
            st.info(f"🔹 {precaution}")

        logger.info(
            "Medicine recommendation displayed for disease: '%s'.",
            recommendation["disease"],
        )

    except MedicineRecommendationError as error:
        st.error(f"Medicine recommendation failed: {error}")
        logger.error("Medicine recommendation failed: %s", error)
    except Exception as error:  # noqa: BLE001
        st.error(
            f"An unexpected error occurred while fetching medicine "
            f"recommendations: {error}"
        )
        logger.exception("Unexpected error during medicine recommendation.")

    st.warning(
        "⚕️ **Medical Disclaimer:** This information is for educational "
        "purposes only and does not constitute medical advice, diagnosis, "
        "dosage guidance, or a prescription. Always consult a qualified "
        "healthcare professional before taking any medication."
    )

    st.divider()

def render_lab_test_section() -> None:
    """Render the recommended lab tests section.

    Reads the previously predicted disease from ``st.session_state`` and
    displays commonly recommended laboratory tests and their purpose for
    educational purposes only.
    """
    st.header("🧪 Recommended Lab Tests")

    predicted_disease = st.session_state.get("predicted_disease")

    if not predicted_disease:
        st.warning(
            "⚠️ No predicted disease found. Please predict a disease in "
            "the **Disease Prediction** section first."
        )
        st.divider()
        return

    try:
        recommendation = get_lab_test_recommendation(predicted_disease)

        st.success(
            f"✅ Lab test recommendations retrieved for "
            f"**{recommendation['disease']}**."
        )

        st.subheader("Disease")
        st.write(recommendation["disease"])

        st.subheader("Recommended Tests")
        tests_markdown = "\n".join(
            f"- {test}" for test in recommendation["recommended_tests"]
        )
        st.markdown(tests_markdown)

        st.subheader("Purpose of Each Test")
        for purpose in recommendation["purpose"]:
            st.info(f"🔹 {purpose}")

        logger.info(
            "Lab test recommendation displayed for disease: '%s'.",
            recommendation["disease"],
        )

    except LabTestRecommendationError as error:
        st.error(f"Lab test recommendation failed: {error}")
        logger.error("Lab test recommendation failed: %s", error)
    except Exception as error:  # noqa: BLE001
        st.error(
            f"An unexpected error occurred while fetching lab test "
            f"recommendations: {error}"
        )
        logger.exception("Unexpected error during lab test recommendation.")

    st.warning(
        "⚕️ **Disclaimer:** These lab test recommendations are for "
        "educational purposes only and do not constitute a medical "
        "diagnosis or treatment plan. Always consult a qualified "
        "healthcare professional for actual testing and diagnosis."
    )

    st.divider()
def render_history_section() -> None:
    """Render the patient disease history section.

    Displays all saved diagnosis history records from ``st.session_state``
    and provides an option to clear the history.
    """
    st.header("📋 Patient Disease History")

    history = st.session_state.get("history", [])

    if not history:
        st.info("ℹ️ No history records available yet.")
        st.divider()
        return

    try:
        for index, record in enumerate(reversed(history), start=1):
            with st.expander(
                f"{index}. {record.get('patient_name', 'Unknown')} — "
                f"{record.get('disease', 'Unknown')}"
            ):
                st.write(f"**Patient Name:** {record.get('patient_name', '-')}")
                st.write(f"**Disease:** {record.get('disease', '-')}")
                st.write(f"**Symptoms:** {', '.join(record.get('symptoms', []))}")
                st.write(f"**Date:** {record.get('date', '-')}")
                st.write(f"**Time:** {record.get('time', '-')}")

        if st.button("🗑️ Clear History"):
            st.session_state["history"] = clear_history(history)
            st.success("✅ History cleared successfully!")
            logger.info("Patient disease history cleared by user.")

    except HistoryError as error:
        st.error(f"History operation failed: {error}")
        logger.error("History operation failed: %s", error)
    except Exception as error:  # noqa: BLE001
        st.error(f"An unexpected error occurred while displaying history: {error}")
        logger.exception("Unexpected error while displaying history.")

    st.divider()


def load_symptom_columns(data_path: str) -> list[str]:
    """Load symptom column names from the training dataset.

    Args:
        data_path: Path to the training dataset CSV file.

    Returns:
        A list of symptom column names, excluding the target column.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset is empty or malformed.
    """
    training_data = pd.read_csv(data_path)

    if training_data.empty:
        raise ValueError(f"Training data at '{data_path}' is empty.")

    symptom_columns = [
        column for column in training_data.columns if column != TARGET_COLUMN
    ]

    if not symptom_columns:
        raise ValueError("No symptom columns found in training data.")

    return symptom_columns


def render_prediction_section() -> None:
    """Render the disease prediction section and handle prediction actions."""
    st.header("🔍 Disease Prediction")
    st.write("Select your symptoms below to predict the possible disease.")

    try:
        symptom_columns = load_symptom_columns(TRAINING_DATA_PATH)
    except FileNotFoundError:
        st.error(f"Training data file not found at: {TRAINING_DATA_PATH}")
        logger.error("Training data file not found at: %s", TRAINING_DATA_PATH)
        return
    except ValueError as error:
        st.error(f"Failed to load symptom list: {error}")
        logger.error("Failed to load symptom list: %s", error)
        return
    except Exception as error:  # noqa: BLE001
        st.error(f"An unexpected error occurred while loading symptoms: {error}")
        logger.exception("Unexpected error while loading symptoms.")
        return

    selected_symptoms = st.multiselect(
        "Search and select your symptoms:",
        options=symptom_columns,
        help="Start typing to search for a symptom.",
    )

    if st.button("Predict Disease"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom before predicting.")
            return

        with st.spinner("Predicting disease, please wait..."):
            try:
                symptoms_dict = {
                    symptom: (1 if symptom in selected_symptoms else 0)
                    for symptom in symptom_columns
                }
                predicted_disease = predict_disease(symptoms_dict)
                st.session_state["predicted_disease"] = predicted_disease

                try:
                    patient = st.session_state.get("patient")
                    patient_name = patient.full_name if patient else "Unknown Patient"

                    history_record = create_history_record(
                        patient_name=patient_name,
                        disease=predicted_disease,
                        symptoms=selected_symptoms,
                    )

                    if "history" not in st.session_state:
                        st.session_state["history"] = []

                    st.session_state["history"] = append_history(
                        st.session_state["history"], history_record
                    )
                    logger.info(
                        "History record saved for patient '%s'.", patient_name
                    )
                except HistoryError as history_error:
                    st.warning(f"Could not save history record: {history_error}")
                    logger.error("Failed to save history record: %s", history_error)

                st.success("✅ Prediction completed successfully!")
                st.subheader("Predicted Disease")
                st.write(predicted_disease)
                logger.info("Predicted disease: %s", predicted_disease)
            except PredictionError as error:
                st.error(f"Prediction failed: {error}")
                logger.error("Prediction failed: %s", error)
            except Exception as error:  # noqa: BLE001
                st.error(
                    f"An unexpected error occurred during prediction: {error}"
                )
                logger.exception("Unexpected error during prediction.")


def main() -> None:
    """Run the Streamlit application."""
    configure_page()
    render_header()
    render_sidebar()
    render_patient_information_section()
    render_bmi_section()
    render_training_section()
    render_prediction_section()
    render_medicine_recommendation_section()
    render_lab_test_section()
    render_history_section()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
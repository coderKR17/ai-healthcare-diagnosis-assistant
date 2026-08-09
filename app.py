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
from pathlib import Path

from src.model import ModelTrainingError, train_model
from src.predictor import PredictionError, predict_disease
from src.patient import Patient, PatientValidationError
from src.ui_style import apply_premium_style
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
from src.pdf_report import (
    generate_medical_report,
    PDFReportError,
)
from src.dashboard import (
    generate_dashboard_summary,
    DashboardError,
)
from src.doctor_recommendation import (
    get_doctor_recommendation,
    DoctorRecommendationError,
)
from src.appointment import (
    create_appointment,
    append_appointment,
    clear_appointments,
    AppointmentError,
)
from src.health_tips import (
    get_health_tips,
    HealthTipsError,
)
from src.auth import (
    register_user,
    login_user,
    logout_user,
    AuthenticationError,
)
from src.email_report import (
    send_medical_report,
    EmailReportError,
)
from src.hospital_finder import (
    get_hospital_recommendation,
    HospitalFinderError,
)
from src.health_chatbot import (
    get_chatbot_response,
    HealthChatbotError,
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

def render_authentication_section() -> None:
    """Render the user authentication section.

    Maintains login state in ``st.session_state`` (``logged_in`` and
    ``current_user`` only) and provides Register/Login tabs, plus a
    welcome message and logout option once authenticated. User accounts
    are persisted permanently by src.auth in data/users.json.
    """
    st.header("🔐 User Authentication")

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "current_user" not in st.session_state:
        st.session_state["current_user"] = None

    if st.session_state["logged_in"]:
        st.success(f"✅ Welcome back, **{st.session_state['current_user']}**!")
        st.info(f"Logged in as: {st.session_state['current_user']}")

        if st.button("Logout"):
            try:
                logout_user()
                st.session_state["logged_in"] = False
                st.session_state["current_user"] = None
                st.success("✅ Logged out successfully!")
                logger.info("User logged out and session state reset.")
                st.rerun()
            except Exception as error:  # noqa: BLE001
                st.error(f"An unexpected error occurred during logout: {error}")
                logger.exception("Unexpected error during logout.")

        st.divider()
        return

    register_tab, login_tab = st.tabs(["Register", "Login"])

    with register_tab:
        st.subheader("Create a New Account")
        register_username = st.text_input("Username", key="register_username")
        register_password = st.text_input(
            "Password", type="password", key="register_password"
        )

        if st.button("Register"):
            try:
                register_user(
                    username=register_username,
                    password=register_password,
                )
                st.success(
                    f"✅ Account created successfully for "
                    f"'{register_username.strip()}'. Please log in."
                )
                logger.info(
                    "User '%s' registered successfully.", register_username.strip()
                )
            except AuthenticationError as error:
                st.error(f"Registration failed: {error}")
                logger.error("Registration failed: %s", error)
            except Exception as error:  # noqa: BLE001
                st.error(f"An unexpected error occurred during registration: {error}")
                logger.exception("Unexpected error during registration.")

    with login_tab:
        st.subheader("Login to Your Account")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input(
            "Password", type="password", key="login_password"
        )

        if st.button("Login"):
            try:
                is_authenticated = login_user(
                    username=login_username,
                    password=login_password,
                )

                if is_authenticated:
                    st.session_state["logged_in"] = True
                    st.session_state["current_user"] = login_username.strip()
                    st.success(f"✅ Welcome, {login_username.strip()}!")
                    logger.info(
                        "User '%s' logged in successfully.", login_username.strip()
                    )
            except AuthenticationError as error:
                st.error(f"Login failed: {error}")
                logger.error("Login failed: %s", error)
            except Exception as error:  # noqa: BLE001
                st.error(f"An unexpected error occurred during login: {error}")
                logger.exception("Unexpected error during login.")

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

def render_pdf_report_section() -> None:
    """Render the PDF medical report generation and download section.

    Reads patient, BMI, predicted disease, medicine, lab test, and history
    data from ``st.session_state``, generates a consolidated PDF medical
    report, and offers it to the user as a downloadable file.
    """
    st.header("📄 Download Medical Report")

    patient = st.session_state.get("patient")
    predicted_disease = st.session_state.get("predicted_disease")
    history = st.session_state.get("history", [])

    if patient is None:
        st.warning(
            "⚠️ No patient information found. Please fill out the "
            "**Patient Information** section first."
        )
        st.divider()
        return

    if not predicted_disease:
        st.warning(
            "⚠️ No predicted disease found. Please predict a disease in "
            "the **Disease Prediction** section first."
        )
        st.divider()
        return

    if st.button("Generate Medical Report"):
        with st.spinner("Generating medical report, please wait..."):
            try:
                bmi_value = calculate_bmi(
                    height_cm=patient.height_cm, weight_kg=patient.weight_kg
                )
                bmi_category = get_bmi_category(bmi_value)
                bmi_data = {
                    "bmi": bmi_value,
                    "category": bmi_category,
                    "health_risk": get_health_risk(bmi_value),
                    "health_tip": get_health_tip(bmi_category),
                }

                medicine_data = get_medicine_recommendation(predicted_disease)
                lab_test_data = get_lab_test_recommendation(predicted_disease)

                output_path = generate_medical_report(
                    patient=patient,
                    bmi_data=bmi_data,
                    predicted_disease=predicted_disease,
                    medicine_data=medicine_data,
                    lab_test_data=lab_test_data,
                    history=history,
                    output_path="reports/medical_report.pdf",
                )

                st.success("✅ Medical report generated successfully!")
                logger.info("Medical report generated at: %s", output_path)

                with open(output_path, "rb") as report_file:
                    st.download_button(
                        label="⬇️ Download Medical Report",
                        data=report_file.read(),
                        file_name="medical_report.pdf",
                        mime="application/pdf",
                    )

            except MedicineRecommendationError as error:
                st.error(f"Medicine recommendation failed: {error}")
                logger.error("Medicine recommendation failed: %s", error)
            except LabTestRecommendationError as error:
                st.error(f"Lab test recommendation failed: {error}")
                logger.error("Lab test recommendation failed: %s", error)
            except BMICalculationError as error:
                st.error(f"BMI calculation failed: {error}")
                logger.error("BMI calculation failed: %s", error)
            except PDFReportError as error:
                st.error(f"Report generation failed: {error}")
                logger.error("Report generation failed: %s", error)
            except Exception as error:  # noqa: BLE001
                st.error(
                    f"An unexpected error occurred while generating the "
                    f"report: {error}"
                )
                logger.exception("Unexpected error during report generation.")

    st.divider()

def render_dashboard_section() -> None:
    """Render the health dashboard summary section.

    Reads patient, predicted disease, and history data from
    ``st.session_state``, calculates BMI, generates a consolidated
    dashboard summary, and displays it using metric cards.
    """
    st.header("📊 Health Dashboard")

    patient = st.session_state.get("patient")
    predicted_disease = st.session_state.get("predicted_disease")
    history = st.session_state.get("history", [])

    if patient is None:
        st.warning(
            "⚠️ No patient information found. Please fill out the "
            "**Patient Information** section first."
        )
        st.divider()
        return

    if not predicted_disease:
        st.warning(
            "⚠️ No predicted disease found. Please predict a disease in "
            "the **Disease Prediction** section first."
        )
        st.divider()
        return

    try:
        bmi_value = calculate_bmi(
            height_cm=patient.height_cm, weight_kg=patient.weight_kg
        )
        bmi_category = get_bmi_category(bmi_value)

        summary = generate_dashboard_summary(
            patient=patient,
            bmi_data={"bmi": bmi_value, "category": bmi_category},
            predicted_disease=predicted_disease,
            history=history,
        )

        st.success(f"✅ Dashboard summary generated for **{summary['patient_name']}**.")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Patient Name", value=summary["patient_name"])
            st.metric(label="Age", value=summary["age"])
        with col2:
            st.metric(label="Gender", value=summary["gender"])
            st.metric(label="BMI", value=f"{summary['bmi']:.2f}")
        with col3:
            st.metric(label="BMI Category", value=summary["bmi_category"])
            st.metric(label="Current Disease", value=summary["predicted_disease"])
        with col4:
            st.metric(label="Total History Records", value=summary["total_history"])

        logger.info(
            "Dashboard summary displayed for patient '%s'.",
            summary["patient_name"],
        )

    except BMICalculationError as error:
        st.error(f"BMI calculation failed: {error}")
        logger.error("BMI calculation failed: %s", error)
    except DashboardError as error:
        st.error(f"Dashboard summary generation failed: {error}")
        logger.error("Dashboard summary generation failed: %s", error)
    except Exception as error:  # noqa: BLE001
        st.error(f"An unexpected error occurred while generating the dashboard: {error}")
        logger.exception("Unexpected error during dashboard generation.")

    st.divider()

def render_doctor_recommendation_section() -> None:
    """Render the doctor recommendation section.

    Reads the previously predicted disease from ``st.session_state`` and
    displays the recommended doctor specialization, department, urgency
    level, emergency status, and general advice.
    """
    st.header("👨‍⚕️ Doctor Recommendation")

    predicted_disease = st.session_state.get("predicted_disease")

    if not predicted_disease:
        st.warning(
            "⚠️ No predicted disease found. Please predict a disease in "
            "the **Disease Prediction** section first."
        )
        st.divider()
        return

    try:
        recommendation = get_doctor_recommendation(predicted_disease)

        st.success(
            f"✅ Doctor recommendation retrieved for "
            f"**{recommendation['disease']}**."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Disease", value=recommendation["disease"])
        with col2:
            st.metric(
                label="Doctor Specialization",
                value=recommendation["doctor_specialization"],
            )
        with col3:
            st.metric(label="Hospital Department", value=recommendation["department"])

        col4, col5 = st.columns(2)
        with col4:
            st.metric(label="Urgency Level", value=recommendation["urgency"])
        with col5:
            st.metric(
                label="Emergency Status",
                value="Yes" if recommendation["emergency"] else "No",
            )

        st.info(f"💡 **General Advice:** {recommendation['advice']}")

        if recommendation["emergency"]:
            st.error(
                "🚨 **EMERGENCY:** This condition may require immediate "
                "medical attention. Please seek emergency care or visit "
                "the nearest hospital right away."
            )

        logger.info(
            "Doctor recommendation displayed for disease: '%s' "
            "(emergency=%s).",
            recommendation["disease"],
            recommendation["emergency"],
        )

    except DoctorRecommendationError as error:
        st.error(f"Doctor recommendation failed: {error}")
        logger.error("Doctor recommendation failed: %s", error)
    except Exception as error:  # noqa: BLE001
        st.error(
            f"An unexpected error occurred while fetching the doctor "
            f"recommendation: {error}"
        )
        logger.exception("Unexpected error during doctor recommendation.")

    st.divider()

def render_appointment_section() -> None:
    """Render the appointment scheduling section.

    Reads patient and predicted disease data from ``st.session_state``,
    fetches a doctor recommendation, and allows the user to schedule and
    manage appointments.
    """
    st.header("📅 Appointment Scheduler")

    patient = st.session_state.get("patient")
    predicted_disease = st.session_state.get("predicted_disease")

    if patient is None:
        st.warning(
            "⚠️ No patient information found. Please fill out the "
            "**Patient Information** section first."
        )
        st.divider()
        return

    if not predicted_disease:
        st.warning(
            "⚠️ No predicted disease found. Please predict a disease in "
            "the **Disease Prediction** section first."
        )
        st.divider()
        return

    try:
        recommendation = get_doctor_recommendation(predicted_disease)
    except DoctorRecommendationError as error:
        st.error(f"Doctor recommendation failed: {error}")
        logger.error("Doctor recommendation failed: %s", error)
        st.divider()
        return
    except Exception as error:  # noqa: BLE001
        st.error(f"An unexpected error occurred while fetching the doctor recommendation: {error}")
        logger.exception("Unexpected error during doctor recommendation lookup.")
        st.divider()
        return

    with st.form(key="appointment_form"):
        st.text_input("Patient Name", value=patient.full_name, disabled=True)
        st.text_input(
            "Doctor Specialization",
            value=recommendation["doctor_specialization"],
            disabled=True,
        )
        st.text_input("Department", value=recommendation["department"], disabled=True)

        appointment_date = st.date_input("Appointment Date")
        appointment_time = st.time_input("Appointment Time")
        notes = st.text_area("Notes", placeholder="Optional additional notes")

        submitted = st.form_submit_button("Schedule Appointment")

    if submitted:
        try:
            appointment = create_appointment(
                patient_name=patient.full_name,
                doctor_specialization=recommendation["doctor_specialization"],
                department=recommendation["department"],
                appointment_date=appointment_date.strftime("%Y-%m-%d"),
                appointment_time=appointment_time.strftime("%H:%M"),
                notes=notes,
            )

            if "appointments" not in st.session_state:
                st.session_state["appointments"] = []

            st.session_state["appointments"] = append_appointment(
                st.session_state["appointments"], appointment
            )

            st.success("✅ Appointment scheduled successfully!")
            st.subheader("Appointment Summary")
            st.write(f"**Patient Name:** {appointment['patient_name']}")
            st.write(f"**Doctor Specialization:** {appointment['doctor_specialization']}")
            st.write(f"**Department:** {appointment['department']}")
            st.write(f"**Date:** {appointment['appointment_date']}")
            st.write(f"**Time:** {appointment['appointment_time']}")
            st.write(f"**Notes:** {appointment['notes'] or '-'}")

            logger.info(
                "Appointment scheduled for patient '%s' on %s at %s.",
                appointment["patient_name"],
                appointment["appointment_date"],
                appointment["appointment_time"],
            )

        except AppointmentError as error:
            st.error(f"Appointment scheduling failed: {error}")
            logger.error("Appointment scheduling failed: %s", error)
        except Exception as error:  # noqa: BLE001
            st.error(f"An unexpected error occurred while scheduling the appointment: {error}")
            logger.exception("Unexpected error during appointment scheduling.")

    existing_appointments = st.session_state.get("appointments", [])
    if existing_appointments:
        st.subheader("Scheduled Appointments")
        for index, appt in enumerate(existing_appointments, start=1):
            with st.expander(
                f"{index}. {appt.get('patient_name', '-')} — "
                f"{appt.get('appointment_date', '-')} {appt.get('appointment_time', '-')}"
            ):
                st.write(f"**Doctor Specialization:** {appt.get('doctor_specialization', '-')}")
                st.write(f"**Department:** {appt.get('department', '-')}")
                st.write(f"**Notes:** {appt.get('notes', '-') or '-'}")

        if st.button("🗑️ Clear Appointments"):
            try:
                st.session_state["appointments"] = clear_appointments(
                    existing_appointments
                )
                st.success("✅ Appointments cleared successfully!")
                logger.info("Appointments cleared by user.")
            except AppointmentError as error:
                st.error(f"Failed to clear appointments: {error}")
                logger.error("Failed to clear appointments: %s", error)
            except Exception as error:  # noqa: BLE001
                st.error(f"An unexpected error occurred while clearing appointments: {error}")
                logger.exception("Unexpected error while clearing appointments.")

    st.divider()

def render_health_tips_section() -> None:
    """Render the health tips and lifestyle recommendations section.

    Reads patient and predicted disease data from ``st.session_state``,
    calculates BMI and BMI category, and displays personalized diet,
    exercise, sleep, hydration, lifestyle, and additional health tips.
    """
    st.header("🥗 Health Tips & Lifestyle Recommendations")

    patient = st.session_state.get("patient")
    predicted_disease = st.session_state.get("predicted_disease")

    if patient is None:
        st.warning(
            "⚠️ No patient information found. Please fill out the "
            "**Patient Information** section first."
        )
        st.divider()
        return

    if not predicted_disease:
        st.warning(
            "⚠️ No predicted disease found. Please predict a disease in "
            "the **Disease Prediction** section first."
        )
        st.divider()
        return

    try:
        bmi_value = calculate_bmi(
            height_cm=patient.height_cm, weight_kg=patient.weight_kg
        )
        bmi_category = get_bmi_category(bmi_value)

        tips = get_health_tips(
            disease=predicted_disease,
            bmi_category=bmi_category,
        )

        st.success(
            f"✅ Health tips generated for **{predicted_disease}** "
            f"(BMI Category: {bmi_category})."
        )

        st.subheader("🍎 Diet Recommendations")
        st.markdown("\n".join(f"- {item}" for item in tips["diet"]))

        st.subheader("🏃 Exercise Recommendations")
        st.markdown("\n".join(f"- {item}" for item in tips["exercise"]))

        st.subheader("😴 Sleep Recommendation")
        st.info(tips["sleep"])

        st.subheader("💧 Hydration Recommendation")
        st.info(tips["hydration"])

        st.subheader("🌿 Lifestyle Advice")
        st.markdown("\n".join(f"- {item}" for item in tips["lifestyle"]))

        st.subheader("📌 Additional Health Tips")
        st.markdown("\n".join(f"- {item}" for item in tips["additional_tips"]))

        logger.info(
            "Health tips displayed for disease '%s' with BMI category '%s'.",
            predicted_disease,
            bmi_category,
        )

    except BMICalculationError as error:
        st.error(f"BMI calculation failed: {error}")
        logger.error("BMI calculation failed: %s", error)
    except HealthTipsError as error:
        st.error(f"Health tips generation failed: {error}")
        logger.error("Health tips generation failed: %s", error)
    except Exception as error:  # noqa: BLE001
        st.error(
            f"An unexpected error occurred while generating health tips: {error}"
        )
        logger.exception("Unexpected error during health tips generation.")

    st.divider()

def render_email_report_section() -> None:
    """Render the email medical report section.

    Reads patient and predicted disease data from ``st.session_state``,
    verifies the generated PDF report exists, and emails it to the
    patient using SMTP credentials provided by the user.
    """
    st.header("📧 Email Medical Report")

    patient = st.session_state.get("patient")
    predicted_disease = st.session_state.get("predicted_disease")

    if patient is None:
        st.warning(
            "⚠️ No patient information found. Please fill out the "
            "**Patient Information** section first."
        )
        st.divider()
        return

    if not predicted_disease:
        st.warning(
            "⚠️ No predicted disease found. Please predict a disease in "
            "the **Disease Prediction** section first."
        )
        st.divider()
        return

    report_path = Path("reports/medical_report.pdf")

    if not report_path.exists():
        st.warning(
            "⚠️ No medical report found. Please generate the report in "
            "the **Download Medical Report** section first."
        )
        st.divider()
        return

    with st.form(key="email_report_form"):
        recipient_email = st.text_input("Recipient Email")
        sender_email = st.text_input("Sender Email")
        sender_password = st.text_input("Sender App Password", type="password")

        submitted = st.form_submit_button("Send Medical Report")

    if submitted:
        with st.spinner("Sending medical report, please wait..."):
            try:
                send_medical_report(
                    recipient_email=recipient_email,
                    pdf_path=str(report_path),
                    sender_email=sender_email,
                    sender_password=sender_password,
                )
                st.success(
                    f"✅ Medical report emailed successfully to "
                    f"{recipient_email.strip()}!"
                )
                logger.info(
                    "Medical report emailed successfully to: %s",
                    recipient_email.strip(),
                )
            except EmailReportError as error:
                st.error(f"Failed to send medical report: {error}")
                logger.error("Failed to send medical report: %s", error)
            except Exception as error:  # noqa: BLE001
                st.error(
                    f"An unexpected error occurred while sending the "
                    f"medical report: {error}"
                )
                logger.exception("Unexpected error while sending medical report.")

    st.divider()

def render_hospital_finder_section() -> None:
    """Render the hospital finder section.

    Reads the previously predicted disease from ``st.session_state`` and
    displays a recommended hospital, including department, address,
    contact information, emergency status, and a Google Maps link.
    """
    st.header("🏥 Hospital Finder")

    predicted_disease = st.session_state.get("predicted_disease")

    if not predicted_disease:
        st.warning(
            "⚠️ No predicted disease found. Please predict a disease in "
            "the **Disease Prediction** section first."
        )
        st.divider()
        return

    try:
        recommendation = get_hospital_recommendation(predicted_disease)

        st.success(
            f"✅ Hospital recommendation retrieved for "
            f"**{recommendation['disease']}**."
        )

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Hospital Name", value=recommendation["hospital_name"])
        with col2:
            st.metric(label="Department", value=recommendation["department"])

        st.info(f"📍 **Address:** {recommendation['address']}")
        st.info(f"📞 **Contact Number:** {recommendation['contact_number']}")

        if recommendation["emergency"]:
            st.error(
                "🚨 **Emergency Hospital Support Available** — This "
                "hospital offers emergency care for this condition."
            )
        else:
            st.info("ℹ️ This hospital does not offer emergency care for this condition.")

        st.link_button("📍 View on Google Maps", recommendation["maps_link"])

        logger.info(
            "Hospital recommendation displayed for disease: '%s' "
            "(emergency=%s).",
            recommendation["disease"],
            recommendation["emergency"],
        )

    except HospitalFinderError as error:
        st.error(f"Hospital recommendation failed: {error}")
        logger.error("Hospital recommendation failed: %s", error)
    except Exception as error:  # noqa: BLE001
        st.error(
            f"An unexpected error occurred while fetching the hospital "
            f"recommendation: {error}"
        )
        logger.exception("Unexpected error during hospital recommendation.")

    st.divider()

def render_health_chatbot_section() -> None:
    """Render the AI health chatbot section.

    Reads the predicted disease and patient data from
    ``st.session_state`` to personalize responses, calculates BMI when
    patient data is available, and lets the user ask general health
    questions to a rule-based (or optionally AI-backed) chatbot.
    """
    st.header("🤖 AI Health Chatbot")

    predicted_disease = st.session_state.get("predicted_disease")
    patient = st.session_state.get("patient")

    bmi_value: float | None = None
    if patient is not None:
        try:
            bmi_value = calculate_bmi(
                height_cm=patient.height_cm, weight_kg=patient.weight_kg
            )
        except BMICalculationError as error:
            st.warning(f"Could not calculate BMI for chatbot context: {error}")
            logger.error("BMI calculation failed in chatbot section: %s", error)
            bmi_value = None

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_message = st.text_area(
        "Ask a health question",
        placeholder="e.g., What should I eat for diabetes?",
        key="chatbot_user_message",
    )

    if st.button("Ask Health Assistant"):
        if not user_message or not user_message.strip():
            st.warning("⚠️ Please enter a question before submitting.")
        else:
            try:
                response = get_chatbot_response(
                    user_message=user_message,
                    predicted_disease=predicted_disease,
                    bmi=bmi_value,
                )

                st.session_state["chat_history"].append(
                    {"user": user_message.strip(), "assistant": response}
                )

                st.success("✅ Response generated successfully!")
                st.info(response)

                logger.info("Chatbot response generated successfully.")

            except HealthChatbotError as error:
                st.error(f"Chatbot could not process your question: {error}")
                logger.error("Chatbot response generation failed: %s", error)
            except Exception as error:  # noqa: BLE001
                st.error(
                    f"An unexpected error occurred while generating the "
                    f"chatbot response: {error}"
                )
                logger.exception("Unexpected error during chatbot response generation.")

    chat_history = st.session_state.get("chat_history", [])
    if chat_history:
        st.subheader("Conversation History")
        for index, entry in enumerate(reversed(chat_history), start=1):
            with st.expander(f"{index}. {entry.get('user', '-')[:60]}"):
                st.write(f"**You:** {entry.get('user', '-')}")
                st.write(f"**Assistant:** {entry.get('assistant', '-')}")

    st.warning(
        "⚕️ **Medical Disclaimer:** This chatbot provides general "
        "educational information only and does not offer a medical "
        "diagnosis. Any medicine information shared is not a "
        "prescription. Please consult a qualified healthcare "
        "professional for personalized advice. If you are experiencing "
        "emergency symptoms, seek immediate professional medical care."
    )

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
    apply_premium_style()
    render_header()
    render_authentication_section()

    if not st.session_state.get("logged_in",False):
        st.info("Please login to access the Healthcare Diagnosis Assistant.")
        return
    render_sidebar()
    render_patient_information_section()
    render_bmi_section()
    render_training_section()
    render_prediction_section()
    render_medicine_recommendation_section()
    render_lab_test_section()
    render_history_section()
    render_pdf_report_section()
    render_dashboard_section()
    render_doctor_recommendation_section()
    render_appointment_section()
    render_health_tips_section()
    render_email_report_section()
    render_hospital_finder_section()
    render_health_chatbot_section()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
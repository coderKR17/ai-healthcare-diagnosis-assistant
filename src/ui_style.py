from __future__ import annotations

import logging

import streamlit as st

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


_PREMIUM_CSS: str = """
<style>

    /* =========================================================
       GLOBAL APP
       ========================================================= */

    .stApp {
        background: #f5f8fb;
        color: #1e293b;
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* =========================================================
       HEADINGS
       ========================================================= */

    h1, h2, h3, h4 {
        color: #1e293b;
        font-weight: 700;
    }

    h1 {
        letter-spacing: -0.5px;
    }

    /* =========================================================
       SIDEBAR
       ========================================================= */

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #2563a8;
    }

    /* =========================================================
       BUTTONS
       ========================================================= */

    .stButton > button {
        background: #2563a8;
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.1rem;
        font-weight: 600;
        transition: all 0.2s ease;
        min-height: 42px;
    }

    .stButton > button:hover {
        background: #1e4e86;
        color: #ffffff;
        border: none;
        transform: translateY(-1px);
    }

    .stButton > button:focus {
        box-shadow: 0 0 0 2px rgba(37, 99, 168, 0.25);
    }

    /* =========================================================
       INPUTS
       ========================================================= */

    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input {
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        background: #ffffff;
    }

    .stTextInput input:focus,
    .stTextArea textarea:focus,
    .stNumberInput input:focus {
        border-color: #2563a8;
        box-shadow: 0 0 0 1px #2563a8;
    }

    /* =========================================================
       SELECT BOXES
       ========================================================= */

    div[data-baseweb="select"] > div {
        border-radius: 10px;
        border-color: #cbd5e1;
        background: #ffffff;
    }

    /* =========================================================
       METRIC CARDS
       ========================================================= */

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b;
    }

    div[data-testid="stMetricValue"] {
        color: #2563a8;
        font-weight: 700;
    }

    /* =========================================================
       ALERT / MESSAGE BOXES
       ========================================================= */

    div[data-testid="stAlert"] {
        border-radius: 12px;
        border-width: 1px;
    }

    /* =========================================================
       EXPANDERS
       ========================================================= */

    details {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        margin-bottom: 0.75rem;
    }

    details summary {
        font-weight: 600;
        color: #1e293b;
    }

    /* =========================================================
       TABS
       ========================================================= */

    button[data-baseweb="tab"] {
        font-weight: 600;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #2563a8;
    }

    /* =========================================================
       DIVIDERS
       ========================================================= */

    hr {
        border-color: #e2e8f0;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }

    /* =========================================================
       FORM CONTAINERS
       ========================================================= */

    div[data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.25rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    /* =========================================================
       LINKS
       ========================================================= */

    a {
        color: #2563a8;
        font-weight: 600;
    }

    /* =========================================================
       RESPONSIVE DESIGN
       ========================================================= */

    @media (max-width: 768px) {

        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }

        h1 {
            font-size: 1.8rem;
        }

        h2 {
            font-size: 1.4rem;
        }

        .stButton > button {
            width: 100%;
        }

        div[data-testid="stMetric"] {
            padding: 0.8rem;
        }
    }

</style>
"""


def apply_premium_style() -> None:
    """Apply premium healthcare dashboard styling to the Streamlit app."""
    try:
        st.markdown(_PREMIUM_CSS, unsafe_allow_html=True)
        logger.info("Premium UI style applied successfully.")
    except Exception:
        logger.exception(
            "Failed to apply premium UI style. Continuing without it."
        )
from __future__ import annotations

import logging

import streamlit as st

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


_PREMIUM_CSS: str = """
<style>

:root {
    --hc-primary: #2563A8;
    --hc-primary-hover: #1E4E86;
    --hc-accent: #0FA3A3;
    --hc-bg: #F5F8FB;
    --hc-surface: #FFFFFF;
    --hc-border: #E2E8F0;
    --hc-text: #1E293B;
    --hc-muted: #64748B;
    --hc-success: #16A34A;
    --hc-warning: #D97706;
    --hc-danger: #DC2626;
    --hc-radius: 10px;
    --hc-radius-sm: 7px;
}

/* Main application background */
.stApp {
    background-color: var(--hc-bg);
    color: var(--hc-text);
    overflow-x: hidden;
}

/* Main content */
.main {
    color: var(--hc-text);
}

/* Headings */
.main h1,
.main h2,
.main h3,
.main h4 {
    color: var(--hc-text) !important;
}

/* Normal text */
.main p,
.main label,
.main span {
    color: var(--hc-text);
}

/* Input fields */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
    caret-color: #2563A8 !important;
}

/* Input placeholder text */
.stTextInput input::placeholder,
.stNumberInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #64748B !important;
    opacity: 1 !important;
}

/* Select boxes */
[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
}

[data-baseweb="select"] span {
    color: #1E293B !important;
}

/* Text area */
.stTextArea textarea {
    min-height: 90px;
}

/* Checkboxes */
[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] label span {
    color: #1E293B !important;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background-color: var(--hc-primary);
    color: #FFFFFF !important;
    border: 1px solid var(--hc-primary);
    border-radius: 8px;
    font-weight: 600;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: var(--hc-primary-hover);
    color: #FFFFFF !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #FFFFFF;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.25rem;
    padding-left: 0.75rem;
    padding-right: 0.75rem;
}

[data-testid="stSidebar"] h1 {
    color: var(--hc-text) !important;
    font-size: 1.3rem;
    border-bottom: 2px solid var(--hc-border);
    padding-bottom: 0.6rem;
    margin-bottom: 0.9rem;
}

[data-testid="stSidebar"] h2 {
    color: var(--hc-primary) !important;
    font-size: 1.02rem;
    margin-top: 1.4rem;
    margin-bottom: 0.5rem;
    padding-top: 0.9rem;
    border-top: 1px solid var(--hc-border);
}

[data-testid="stSidebar"] hr {
    border-top: 1px solid var(--hc-border);
}

/* Section headers */
.main h2 {
    display: block;
    background-color: var(--hc-surface);
    color: var(--hc-text) !important;
    border: 1px solid var(--hc-border);
    border-left: 4px solid var(--hc-primary);
    border-radius: var(--hc-radius-sm);
    padding: 0.65rem 1rem;
    margin-top: 2rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
}

/* Tabs */
[data-baseweb="tab-list"] {
    background-color: #FFFFFF;
    border: 1px solid var(--hc-border);
    border-radius: var(--hc-radius-sm);
    padding: 0.25rem;
    gap: 0.25rem;
}

[data-baseweb="tab"] {
    color: var(--hc-text) !important;
    border-radius: 6px;
    padding: 0.5rem 1rem;
}

[aria-selected="true"][data-baseweb="tab"] {
    background-color: var(--hc-bg);
    color: var(--hc-primary) !important;
    box-shadow: inset 0 0 0 1px var(--hc-primary);
}

/* Forms / Cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border: 1px solid var(--hc-border) !important;
    border-radius: var(--hc-radius) !important;
    padding: 0.5rem 0.25rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
    margin-bottom: 1rem;
}

/* Alerts */
[data-testid="stAlert"] {
    color: var(--hc-text);
}

/* Mobile */
@media (max-width: 768px) {

    [data-testid="stSidebar"] > div:first-child {
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }

    .main h2 {
        padding: 0.5rem 0.75rem;
        margin-top: 1.4rem;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        width: 100%;
    }

    [data-testid="column"] {
        min-width: 100% !important;
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
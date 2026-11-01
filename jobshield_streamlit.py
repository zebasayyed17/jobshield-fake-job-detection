import streamlit as st
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="Fake Job Posting Detector",
    page_icon="🔍",
    layout="wide"
)

# -------------------------------
# VISUAL DESIGN (UI ONLY)
# -------------------------------
st.markdown("""
<style>
    :root {
        --ink: #172b27;
        --muted: #65736e;
        --green: #176b52;
        --green-dark: #104b3b;
        --mint: #eaf5ef;
        --line: #dce7e1;
        --surface: #ffffff;
        --canvas: #f5f8f6;
    }

    .stApp {
        background:
            radial-gradient(circle at 90% 0%, rgba(207, 235, 220, 0.42), transparent 28%),
            var(--canvas);
        color: var(--ink);
    }

    [data-testid="stHeader"] { background: transparent; }
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }

    h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] {
        color: var(--ink);
    }

    .hero {
        padding: 1.7rem 1.8rem;
        border-radius: 22px;
        background: linear-gradient(120deg, #123f33 0%, #176b52 70%, #278265 100%);
        color: white;
        margin-bottom: 1.1rem;
        box-shadow: 0 12px 28px rgba(19, 73, 55, 0.13);
    }
    .hero .eyebrow {
        color: #c5e8d7;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .hero h1 {
        color: white;
        font-size: clamp(2rem, 4vw, 3rem);
        line-height: 1.12;
        margin: 0 0 0.7rem 0;
        font-weight: 750;
    }
    .hero p {
        color: #e6f4ec;
        font-size: 1.02rem;
        line-height: 1.6;
        margin: 0;
        max-width: 820px;
    }

    .how-it-works {
        padding: 1rem 1.2rem;
        background: #eaf5ef;
        border: 1px solid #d4e9dc;
        border-radius: 14px;
        margin: 0.9rem 0 1.2rem 0;
        color: #214b3b;
        line-height: 1.55;
    }
    .how-it-works strong { color: #104b3b; }

    [data-testid="stForm"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 1.35rem 1.5rem 1.5rem 1.5rem;
        box-shadow: 0 8px 24px rgba(24, 57, 43, 0.045);
    }

    h2 { font-size: 1.55rem !important; font-weight: 700 !important; }
    h3 { font-size: 1.1rem !important; font-weight: 650 !important; }

    [data-testid="stTextInput"] label,
    [data-testid="stTextArea"] label,
    [data-testid="stSelectbox"] label {
        font-weight: 600;
        color: #30443d;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: #f8faf9;
        border: 1px solid #dbe5df;
        border-radius: 10px;
    }
    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {
        border-color: #278265;
        box-shadow: 0 0 0 1px #278265;
    }

    .stButton > button,
    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(90deg, #176b52, #218060);
        color: white;
        border: 0;
        border-radius: 11px;
        min-height: 3rem;
        font-weight: 700;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        background: #104b3b;
        color: white;
        box-shadow: 0 6px 16px rgba(23, 107, 82, 0.18);
        transform: translateY(-1px);
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1rem 1.1rem;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--green-dark); font-weight: 750; }
    [data-testid="stProgress"] > div > div {
        background-color: #218060;
    }

    [data-testid="stAlert"] { border-radius: 13px; }
    hr { border-color: #dce7e1; }
    .footer-note {
        color: #718078;
        text-align: center;
        font-size: 0.85rem;
        padding-top: 0.3rem;
    }

    @media (max-width: 700px) {
        .block-container { padding: 1rem 0.8rem 2rem 0.8rem; }
        .hero { padding: 1.25rem; border-radius: 16px; }
        [data-testid="stForm"] { padding: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL AND TOKENIZER
# -------------------------------
@st.cache_resource
def load_resources():
    model = load_model("fake_job_rnn.keras")

    with open("fake_job_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer

try:
    model, tokenizer = load_resources()
except Exception as e:
    st.error("Unable to load the model or tokenizer.")
    st.write("Make sure both saved files are in the same folder as app.py.")
    st.exception(e)
    st.stop()

# Same settings as used during training
max_num_words = 10000
seq_len = 50

# -------------------------------
# APP TITLE
# -------------------------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">JOB SAFETY • SMART SCREENING</div>
    <h1>🔎 JobShield</h1>
    <p>Check a job posting for possible warning signs before you apply.
    Add the details you have and let the model screen the listing in seconds.</p>
</div>
<div class="how-it-works">
    <strong>How to use this tool:</strong> Fill in the job details you know,
    then select <strong>Predict Job Authenticity</strong>. You can start with
    the job title and description. The result is an AI-based signal, so always
    verify the company and offer independently.
</div>
""", unsafe_allow_html=True)

# -------------------------------
# JOB DETAILS FORM
# -------------------------------
with st.form("job_form"):

    st.subheader("Tell us about the job")
    st.caption("Only enter information shown in the job advertisement. Leave unavailable details blank.")

    title = st.text_input("Job Title")
    location = st.text_input("Location")
    department = st.text_input("Department")
    salary_range = st.text_input("Salary Range")

    company_profile = st.text_area(
        "Company Profile",
        height=100
    )

    description = st.text_area(
        "Job Description",
        height=150
    )

    requirements = st.text_area(
        "Job Requirements",
        height=120
    )

    col1, col2 = st.columns(2)

    with col1:
        has_company_logo = st.selectbox(
            "Has Company Logo?",
            ["0", "1", "Unknown"]
        )

        employment_type = st.selectbox(
            "Employment Type",
            [
                "Full-time",
                "Part-time",
                "Contract",
                "Temporary",
                "Other",
                "Unknown"
            ]
        )

    with col2:
        required_experience = st.selectbox(
            "Required Experience",
            [
                "Internship",
                "Entry level",
                "Associate",
                "Mid-Senior level",
                "Director",
                "Executive",
                "Not Applicable",
                "Unknown"
            ]
        )

        industry = st.text_input("Industry")

    submitted = st.form_submit_button(
        "🔍 Predict Job Authenticity",
        use_container_width=True
    )

# -------------------------------
# PREDICTION
# -------------------------------
if submitted:

    if not title.strip() and not description.strip():
        st.warning(
            "Please enter at least a job title or job description."
        )

    else:
        # Match the text construction used in your notebook
        job_text = (
            str(title) + " " +
            str(location) + " " +
            str(department) + " " +
            str(salary_range) + " " +
            str(company_profile) + " " +
            str(company_profile) + " " +
            str(description) +
            str(requirements) + " " +
            str(has_company_logo) + " " +
            str(employment_type) + " " +
            str(required_experience) + " " +
            str(industry)
        )

        # Convert text into tokens
        sequence = tokenizer.texts_to_sequences([job_text])

        # Pad to the same sequence length as training
        padded_sequence = pad_sequences(
            sequence,
            maxlen=seq_len
        )

        # Predict probabilities
        prediction = model.predict(
            padded_sequence,
            verbose=0
        )

        predicted_class = int(np.argmax(prediction, axis=1)[0])

        real_probability = float(prediction[0][0]) * 100
        fake_probability = float(prediction[0][1]) * 100

        st.divider()
        st.subheader("Screening result")

        if predicted_class == 1:
            st.error("⚠️ Prediction: Potentially Fake Job Posting")
            st.write(
                "The model classified this posting as fraudulent. "
                "Verify the employer and job details before proceeding."
            )
        else:
            st.success("✅ Prediction: Likely Genuine Job Posting")
            st.write(
                "The model classified this posting as genuine. "
                "This does not guarantee that the job is legitimate."
            )

        # Display model probabilities
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Genuine Job Probability",
                f"{real_probability:.2f}%"
            )
            st.progress(real_probability / 100)

        with col2:
            st.metric(
                "Fake Job Probability",
                f"{fake_probability:.2f}%"
            )
            st.progress(fake_probability / 100)

        st.caption(
            "Probabilities are model outputs, not verified measures "
            "of the actual likelihood of fraud. Always verify job offers."
        )

# -------------------------------
# FOOTER
# -------------------------------
st.divider()

st.markdown(
    '<div class="footer-note">Built with Python · TensorFlow · NLP · Streamlit'
    '<br>Use this result as a screening aid, not as a guarantee.</div>',
    unsafe_allow_html=True
)
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
st.title("🔍 Fake Job Posting Detection System")

st.write(
    "Analyze job postings using a trained SimpleRNN deep learning model "
    "to identify potentially fraudulent job advertisements."
)

st.info(
    "Enter the available job details below and click Predict "
    "to classify the posting."
)

# -------------------------------
# JOB DETAILS FORM
# -------------------------------
with st.form("job_form"):

    st.subheader("Enter Job Posting Details")

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
        st.subheader("Prediction Results")

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

st.caption(
    "Built with Python, TensorFlow, Keras, NLP preprocessing, "
    "SimpleRNN and Streamlit."
)
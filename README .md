# JobShield — Fake Job Posting Detection

JobShield is a Streamlit application that screens job advertisements for potential warning signs using a trained Keras deep-learning model. Users enter details from a job listing and receive a prediction labelled **Likely Genuine Job Posting** or **Potentially Fake Job Posting**, along with the model's output scores for both classes.

> **Important:** JobShield is a screening aid, not proof that a job is genuine or fraudulent. Always verify the employer, recruiter, company website/domain, and offer details independently.

## Features

- Web form for job title, location, department, salary range, company profile, job description, requirements, company-logo status, employment type, experience level, and industry
- Text preprocessing with a saved tokenizer and sequence padding
- Trained Keras model inference
- Displays genuine-class and fake-class output scores
- Streamlit interface with a forest-green and light background design
- Basic validation requiring a job title or job description before prediction

## Technology Stack

- **Python**
- **Streamlit** — web application interface
- **TensorFlow / Keras** — loading and running the trained model
- **NumPy** — prediction output handling
- **Pickle** — loading the saved tokenizer
- **NLP preprocessing** — tokenization and sequence padding

## Project Files

Place these files in the same project folder:

```text
jobshield-fake-job-detection/
├── app.py
├── fake_job_rnn.keras
├── fake_job_tokenizer.pkl
├── requirements.txt
└── README.md
```

The `.keras` model and `.pkl` tokenizer filenames must match the filenames referenced in `app.py`. If your uploaded app file has a different name, rename it to `app.py` in the GitHub repository or adjust the run command accordingly.

## Run the Project Locally

### 1. Download or clone the repository

After creating the GitHub repository, clone it with:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Replace the placeholder URL with your actual repository URL.

### 2. Create and activate a virtual environment (recommended)

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the Streamlit application

If the application file is named `app.py`:

```bash
streamlit run app.py
```

Keep `fake_job_rnn.keras` and `fake_job_tokenizer.pkl` in the same folder as the application file.

## How It Works

1. The user enters the available information from a job advertisement.
2. The application combines the entered fields into a text input.
3. The saved tokenizer converts the text into a sequence of tokens.
4. The sequence is padded to the configured length of **50 tokens**.
5. The trained Keras model produces output scores.
6. The application displays a class prediction and the two output scores.

## Model and Evaluation Notes

- The current application uses a sequence length of `50`.
- The app code interprets output index `0` as **genuine** and output index `1` as **fake**. **Verify this mapping against the class labels used during model training before relying on the predictions.**
- The tokenizer, sequence length, field order, and text preprocessing must be consistent with the training pipeline.
- The supplied files do not establish the dataset source or license, the train/test split, or verified evaluation results. Add dataset details and metrics (such as precision, recall, F1-score, accuracy, and a confusion matrix) only after checking the original training notebook or evaluation output.
- Model output scores should not be treated as calibrated real-world fraud probabilities.

## Deployment

You can deploy the app with [Streamlit Community Cloud](https://share.streamlit.io/).

1. Push the application, model, tokenizer, and dependency file to GitHub.
2. Open Streamlit Community Cloud and connect your GitHub account.
3. Create a new app and select the repository, branch, and app entry-point file.
4. Deploy the app and review any build errors.
5. Test the deployed app with sample listings before sharing the public link.

**Live demo:** Add the deployed app URL here after deployment.

## Limitations

- The model's performance depends on the data and labels used during training.
- A genuine listing could be flagged, and a fraudulent listing could be missed.
- The application does not independently verify employers, recruiters, domains, or job offers.
- Missing fields or differences between training and prediction preprocessing may affect results.
- The output is not a substitute for independent checks or professional judgement.

## Future Improvements

- Document the dataset source and its usage license.
- Include the training notebook and reproducible preprocessing steps.
- Add verified evaluation metrics and a confusion matrix.
- Add a screenshot or short demo GIF.
- Explore interpretable warning indicators and independent company-verification checks.

## Security Note

The tokenizer is loaded from a pickle file. Pickle files can execute code when deserialized, so only load tokenizer files from a trusted source.

## Author

**Zeba Sayyed**

- **LinkedIn:** https://www.linkedin.com/in/zeba-sayyed-28766941b
- **GitHub:** Add the repository URL after creating it

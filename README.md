# XAI Shield

XAI Shield is a student capstone prototype for an explainable-AI intrusion detection system. The goal is to classify network activity as **normal or malicious** and provide the analyst with information about why the model made that decision.

## Current MVP

- Simple Streamlit cybersecurity dashboard
- Random Forest binary classifier
- User-entered sample network-traffic values
- Normal/malicious classification and confidence
- Feature-importance explanation
- Generated demo training data so the repository runs without downloading a large dataset

> This version is an academic prototype and is not a production intrusion-detection system. SHAP/LIME and a real cybersecurity dataset are planned next steps.

## Requirements

- Python 3.11 or 3.12 recommended
- pip

## Run on Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Run on macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local address, normally `http://localhost:8501`.

## Project Structure

```text
xai-shield/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How it works

1. The application creates a small generated dataset representing network-traffic features.
2. A Random Forest classifier is trained when the application starts.
3. The analyst enters sample traffic values.
4. The model classifies the activity as normal or malicious.
5. The dashboard displays the classification, confidence, and model feature importance.

## Next Development Steps

1. Replace generated demo data with a recognized cybersecurity dataset.
2. Separate training and prediction code.
3. Add SHAP explanations.
4. Add LIME explanations and compare them with SHAP.
5. Improve model evaluation with accuracy, precision, recall, F1 score, and confusion matrix.
6. Simplify and refine the analyst dashboard based on proposal feedback.
7. Test the repository on a second computer before final submission.

## Disclaimer

This software is for academic demonstration and research only. It should not be used as a production security control.

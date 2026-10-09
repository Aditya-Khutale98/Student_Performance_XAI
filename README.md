# Student Performance Predictor Using Explainable AI

## Project Overview
This project predicts a student's expected exam score using Machine Learning and explains the prediction using Explainable AI (XAI).

The system uses three inputs:
- Study Hours per Day
- Attendance (%)
- Previous Performance

It displays the predicted score, feature contributions using SHAP, prediction uncertainty information, and personalized recommendations.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- SHAP (Explainable AI)
- Streamlit
- Joblib

## Key Features
- Predicts an exam score between 0 and 100.
- Accepts study time in HH.MM format (for example, 3.45 means 3 hours and 45 minutes).
- Explains predictions using SHAP feature contributions.
- Displays estimated prediction error and uncertainty information.
- Warns when inputs fall outside the ranges observed in the training dataset.
- Provides recommendations based on the prediction and input factors.

## Project Structure
- `app/` — Streamlit application
- `data/` — Datasets
- `models/` — Trained model and metadata
- `src/` — Training, prediction, validation, and explanation code
- `requirements.txt` — Python dependencies

## How to Run Locally
1. Create and activate a Python virtual environment.
2. Install dependencies:

   `pip install -r requirements.txt`

3. Run the application from the project root:

   `streamlit run app/app.py`

## Important Note
This project is an educational and research prototype. Predictions are estimates, not guaranteed exam results. The training dataset is synthetic, and predictions outside the observed training-data ranges may be less reliable.

## Live Application
[Open Student Performance Predictor](https://studentperformancexai-i9ycrnm4u2ubwvw5u8s2la.streamlit.app/)
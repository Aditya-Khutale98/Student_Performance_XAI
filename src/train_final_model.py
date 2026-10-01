import os
import json
import joblib
import numpy as np
import pandas as pd

from bounded_model import BoundedRegression


# ============================================================
# PATHS
# ============================================================

DATA_PATH = (
    "data/raw/StudentPerformanceFactors.csv"
)

MODEL_DIR = "models"

MODEL_PATH = (
    "models/student_performance_bounded.pkl"
)

METADATA_PATH = (
    "models/student_performance_bounded_metadata.json"
)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    DATA_PATH
)


# ============================================================
# CLEAN TARGET
# ============================================================

# Exam score should be between 0 and 100.
# Dataset contains one value of 101.
df["Exam_Score"] = df["Exam_Score"].clip(
    0,
    100
)


# ============================================================
# CONVERT WEEKLY STUDY HOURS → DAILY STUDY HOURS
# ============================================================

df["Study_Hours_Daily"] = (
    df["Hours_Studied"] / 7.0
)


# ============================================================
# FINAL FEATURES
# ============================================================

FEATURES = [
    "Study_Hours_Daily",
    "Attendance",
    "Previous_Scores"
]

TARGET = "Exam_Score"


X = df[FEATURES].copy()

y = df[TARGET].copy()


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print()

print("=" * 70)

print(
    "TRAINING FINAL STUDENT PERFORMANCE MODEL"
)

print("=" * 70)

print()

print(
    "Students :",
    len(df)
)

print(
    "Features :",
    ", ".join(FEATURES)
)

print(
    "Target   :",
    TARGET
)

print()


# ============================================================
# CREATE AND TRAIN BOUNDED MODEL
# ============================================================

model = BoundedRegression()

model.fit(
    X,
    y
)


# ============================================================
# TEST SAMPLE PREDICTIONS
# ============================================================

test_students = pd.DataFrame(
    [
        [0.0, 0.0, 0.0],
        [2.0, 50.0, 50.0],
        [4.0, 70.0, 60.0],
        [6.0, 80.0, 70.0],
        [8.0, 90.0, 80.0],
        [12.0, 95.0, 90.0],
        [16.0, 100.0, 100.0],
        [20.0, 100.0, 100.0],
        [24.0, 100.0, 100.0],
    ],
    columns=FEATURES
)


predictions = model.predict(
    test_students
)


# ============================================================
# DISPLAY SAMPLE PREDICTIONS
# ============================================================

print()

print(
    "SAMPLE PREDICTIONS"
)

print(
    "-" * 70
)


for i, prediction in enumerate(
    predictions
):

    study = (
        test_students
        .iloc[i]["Study_Hours_Daily"]
    )

    attendance = (
        test_students
        .iloc[i]["Attendance"]
    )

    previous = (
        test_students
        .iloc[i]["Previous_Scores"]
    )

    print(
        f"Study={study:5.2f} | "
        f"Attendance={attendance:6.2f} | "
        f"Previous={previous:6.2f} | "
        f"Prediction={prediction:6.2f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "model_type":
        "Logistic Bounded Regression",

    "dataset":
        "StudentPerformanceFactors.csv",

    "dataset_rows":
        int(len(df)),

    "features":
        FEATURES,

    "target":
        TARGET,

    "target_range":
        [0, 100],

    "study_hours_original":
        "Hours_Studied is weekly study hours",

    "study_hours_conversion":
        "Study_Hours_Daily = Hours_Studied / 7",

    "validation":
        "5-Fold Cross Validation × 5 Repeats",

    "robust_validation_metrics": {

        "MAE_mean":
            1.4429,

        "MAE_std":
            0.0544,

        "RMSE_mean":
            2.5372,

        "RMSE_std":
            0.3101,

        "R2_mean":
            0.5734,

        "R2_std":
            0.0662
    },

    "prediction_constraint":
        "0-100",

    "notes":
        [
            "Final predictions are bounded between 0 and 100.",

            "The model accepts user inputs across "
            "the defined physical input ranges.",

            "Extreme inputs may represent extrapolation "
            "beyond the training data.",

            "The dataset contains one Exam_Score value "
            "of 101, which was clipped to 100."
        ]
}


with open(
    METADATA_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print()

print("=" * 70)

print(
    "FINAL MODEL SAVED"
)

print("=" * 70)

print()

print(
    "Model    :",
    MODEL_PATH
)

print(
    "Metadata :",
    METADATA_PATH
)

print()

print(
    "Training completed successfully."
)

print()

print("=" * 70)
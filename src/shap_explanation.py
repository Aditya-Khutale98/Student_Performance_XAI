import pandas as pd
import joblib
import shap


# ============================================================
# 1. Load trained model
# ============================================================

model = joblib.load(
    "models/student_performance_model.pkl"
)


# ============================================================
# 2. Load original dataset for SHAP background
# ============================================================

data = pd.read_csv(
    "data/raw/student_exam_performance.csv"
)


# ============================================================
# 3. Convert HH.MM study time into decimal hours
# ============================================================

def convert_study_time(time_value):

    time_value = str(time_value).strip()

    if "." in time_value:

        parts = time_value.split(".")

        if len(parts) != 2:
            raise ValueError(
                "Please enter study time in HH.MM format."
            )

        hours = int(parts[0])
        minutes_text = parts[1]

        if len(minutes_text) == 1:
            minutes_text = minutes_text + "0"

        minutes = int(minutes_text)

    else:

        hours = int(time_value)
        minutes = 0

    if hours < 0:
        raise ValueError(
            "Study hours cannot be negative."
        )

    if minutes < 0 or minutes >= 60:
        raise ValueError(
            "Minutes must be between 00 and 59."
        )

    decimal_hours = hours + (minutes / 60)

    return decimal_hours


# ============================================================
# 4. Student Input
# ============================================================

# HH.MM format
#
# 3.45 = 3 hours 45 minutes
# 3.30 = 3 hours 30 minutes
# 4.00 = 4 hours
# 5.15 = 5 hours 15 minutes
#
# Later, the Streamlit application will have
# separate Hours and Minutes inputs.

study_time = 3.45

attendance = 85

sleep_hours = 7


# ============================================================
# 5. Convert study time for the ML model
# ============================================================

study_hours_decimal = convert_study_time(
    study_time
)


print("\nStudy Time:")
print(f"User Input: {study_time:.2f}")

print(
    f"Converted for Model: "
    f"{study_hours_decimal:.2f} decimal hours"
)


# ============================================================
# 6. Prepare features
# ============================================================

features = [
    "hours_studied",
    "attendance_percentage",
    "sleep_hours"
]


# ============================================================
# 7. Prepare SHAP background data
# ============================================================

background_data = data[features].copy()


# ============================================================
# 8. Prepare new student data
# ============================================================

student_data = pd.DataFrame(
    [[
        study_hours_decimal,
        attendance,
        sleep_hours
    ]],
    columns=features
)


# ============================================================
# 9. Make prediction
# ============================================================

prediction = model.predict(
    student_data
)


print("\n" + "=" * 50)
print("STUDENT PERFORMANCE PREDICTION")
print("=" * 50)

print(
    f"\n🎯 Predicted Exam Score: "
    f"{prediction[0]:.2f}%"
)


# ============================================================
# 10. SHAP Explanation
# ============================================================

# Use the original student dataset as the
# background/reference for SHAP.

explainer = shap.Explainer(
    model,
    background_data
)

shap_values = explainer(
    student_data
)


# ============================================================
# 11. Get SHAP values
# ============================================================

study_shap = shap_values.values[0][
    features.index("hours_studied")
]

attendance_shap = shap_values.values[0][
    features.index("attendance_percentage")
]

sleep_shap = shap_values.values[0][
    features.index("sleep_hours")
]


# ============================================================
# 12. Explain the prediction in simple words
# ============================================================

print("\n" + "=" * 50)
print("EXPLAINABLE AI - SIMPLE EXPLANATION")
print("=" * 50)

print(
    f"\n🎯 Predicted Exam Score: "
    f"{prediction[0]:.2f}%"
)

print(
    "\n📊 Why did the model give this prediction?\n"
)


# ------------------------------------------------------------
# Study Time
# ------------------------------------------------------------

if study_shap > 0:

    print(
        "🟢 Study Time: "
        "Your study time had a positive effect "
        "on the prediction."
    )

elif study_shap < 0:

    print(
        "🔴 Study Time: "
        "Your study time had a negative effect "
        "on the prediction."
    )

else:

    print(
        "🟡 Study Time: "
        "Your study time had very little effect "
        "on the prediction."
    )


# ------------------------------------------------------------
# Attendance
# ------------------------------------------------------------

if attendance_shap > 0:

    print(
        "🟢 Attendance: "
        "Your attendance had a positive effect "
        "on the prediction."
    )

elif attendance_shap < 0:

    print(
        "🟡 Attendance: "
        "Your attendance had a negative effect "
        "on the prediction."
    )

else:

    print(
        "🟡 Attendance: "
        "Your attendance had very little effect "
        "on the prediction."
    )


# ------------------------------------------------------------
# Sleep
# ------------------------------------------------------------

if sleep_shap > 0:

    print(
        "🟢 Sleep: "
        "Your sleep duration had a positive effect "
        "on the prediction."
    )

elif sleep_shap < 0:

    print(
        "🔴 Sleep: "
        "Your sleep duration had a negative effect "
        "on the prediction."
    )

else:

    print(
        "🟡 Sleep: "
        "Your sleep duration had very little effect "
        "on the prediction."
    )


# ============================================================
# 13. Simple Suggestions
# ============================================================

suggestions = []


# ------------------------------------------------------------
# Study Time Suggestion
# ------------------------------------------------------------

if study_shap < 0:

    suggestions.append(
        "Consider increasing your daily study time."
    )

elif study_shap > 0:

    suggestions.append(
        "Your study time is positively contributing "
        "to the prediction. Try to maintain this routine."
    )


# ------------------------------------------------------------
# Attendance Suggestion
# ------------------------------------------------------------

if attendance_shap < 0:

    suggestions.append(
        "Try to improve your attendance and attend "
        "classes regularly."
    )

elif attendance_shap > 0:

    suggestions.append(
        "Your attendance is positively contributing "
        "to the prediction. Try to maintain regular attendance."
    )


# ------------------------------------------------------------
# Sleep Suggestion
# ------------------------------------------------------------

if sleep_shap < 0:

    suggestions.append(
        "Try to maintain a healthy and consistent "
        "sleep schedule."
    )

elif sleep_shap > 0:

    suggestions.append(
        "Your sleep duration is positively contributing "
        "to the prediction. Try to maintain a healthy "
        "sleep routine."
    )


# ============================================================
# 14. Display Suggestions
# ============================================================

print("\n💡 Simple Suggestions:\n")


if len(suggestions) == 0:

    print(
        "✅ Your current inputs are not showing any "
        "major areas that require improvement."
    )

else:

    for suggestion in suggestions:

        print(
            "• " + suggestion
        )


# ============================================================
# 15. Display SHAP Contribution Values
# ============================================================

print("\n" + "=" * 50)
print("SHAP CONTRIBUTION VALUES")
print("=" * 50)

print(
    f"\nStudy Time SHAP: "
    f"{study_shap:.2f}"
)

print(
    f"Attendance SHAP: "
    f"{attendance_shap:.2f}"
)

print(
    f"Sleep SHAP: "
    f"{sleep_shap:.2f}"
)
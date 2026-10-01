import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# 1. Load dataset
file_path = "data/raw/student_exam_performance.csv"

data = pd.read_csv(file_path)


# 2. Select input features
X = data[
    [
        "hours_studied",
        "attendance_percentage",
        "sleep_hours"
    ]
]


# 3. Select target
y = data["exam_score"]


# 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 5. Create Linear Regression model
model = LinearRegression()


# 6. Train the model
model.fit(X_train, y_train)


# 7. Save the trained model
joblib.dump(
    model,
    "models/student_performance_model.pkl"
)


print("Model trained successfully!")
print("Model saved successfully!")
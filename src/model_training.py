import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


# 4. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Data preparation completed!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 5. Create the Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# 6. Train the model
model.fit(X_train, y_train)

print("Random Forest model trained successfully!")

# 7. Make predictions on testing data
predictions = model.predict(X_test)

print("\nPredicted Exam Scores:")
print(predictions)

# 8. Evaluate the model

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nRandom Forest Model Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

print("\nActual vs Predicted:")

comparison = pd.DataFrame({
    "Actual Score": y_test.values,
    "Predicted Score": predictions
})

print(comparison.head(10))

# 9. Gradient Boosting Model

gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=2,
    random_state=42
)

gb_model.fit(X_train, y_train)

gb_predictions = gb_model.predict(X_test)

gb_mae = mean_absolute_error(y_test, gb_predictions)
gb_rmse = mean_squared_error(y_test, gb_predictions) ** 0.5
gb_r2 = r2_score(y_test, gb_predictions)

print("\nGradient Boosting Model Evaluation:")
print("MAE:", gb_mae)
print("RMSE:", gb_rmse)
print("R2 Score:", gb_r2)

# 10. Baseline Model - Predict the Average Score

baseline_prediction = y_train.mean()

baseline_predictions = [baseline_prediction] * len(y_test)

baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_rmse = mean_squared_error(y_test, baseline_predictions) ** 0.5
baseline_r2 = r2_score(y_test, baseline_predictions)

print("\nBaseline Model Evaluation:")
print("Average Score:", baseline_prediction)
print("MAE:", baseline_mae)
print("RMSE:", baseline_rmse)
print("R2 Score:", baseline_r2)
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
file_path = "data/raw/student_exam_performance.csv"

data = pd.read_csv(file_path)

print("Dataset Shape:")
print(data.shape)

print("\nColumn Names:")
print(data.columns)

print("\nMissing Values:")
print(data.isnull().sum())

print("\nData Types:")
print(data.dtypes)

print("\nStatistical Summary:")
print(data.describe())

print("\nFeatures (X):")
X = data[["hours_studied", "attendance_percentage", "sleep_hours"]]
print(X.head())

print("\nTarget (y):")
y = data["exam_score"]
print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting Data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Training Completed!")

predictions = model.predict(X_test)

print("\nPredicted Scores:")
print(predictions)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

print("\nCorrelation with Exam Score:")
print(data.corr(numeric_only=True)["exam_score"].sort_values(ascending=False))
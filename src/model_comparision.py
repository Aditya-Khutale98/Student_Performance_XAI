import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
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


# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 5. Create models
models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=100,
        random_state=42
    ),
    "Polynomial Regression": make_pipeline(
        PolynomialFeatures(degree=2),
        LinearRegression()
    )
}


# 6. Train and evaluate each model
results = []


for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


# 7. Display results
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df.to_string(index=False))


# 8. 5-Fold Cross-Validation
print("\n5-Fold Cross-Validation:")
for name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="neg_mean_absolute_error"
    )

    mae_scores = -scores

    print(
        name,
        "Average MAE:",
        mae_scores.mean()
    )
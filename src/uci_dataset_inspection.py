from ucimlrepo import fetch_ucirepo
import pandas as pd


# ============================================================
# UCI STUDENT PERFORMANCE DATASET INSPECTION
# ============================================================

print("\n" + "=" * 70)
print("UCI STUDENT PERFORMANCE DATASET INSPECTION")
print("=" * 70)


# ------------------------------------------------------------
# 1. Download dataset
# ------------------------------------------------------------

data = fetch_ucirepo(id=320)

X = data.data.features.copy()
y = data.data.targets.copy()

print("\nDataset downloaded successfully.")


# ------------------------------------------------------------
# 2. Dataset size
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET SIZE")
print("=" * 70)

print("\nNumber of students:", len(X))
print("Number of features:", X.shape[1])
print("Number of targets:", y.shape[1])


# ------------------------------------------------------------
# 3. Feature columns
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE COLUMNS")
print("=" * 70)

for i, column in enumerate(X.columns, start=1):
    print(f"{i:2d}. {column}")


# ------------------------------------------------------------
# 4. Target columns
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TARGET COLUMNS")
print("=" * 70)

for column in y.columns:
    print("•", column)


# ------------------------------------------------------------
# 5. Data types
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(
    X.dtypes.to_string()
)


# ------------------------------------------------------------
# 6. Missing values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing_features = X.isnull().sum()

missing_targets = y.isnull().sum()

print("\nFeatures:")
print(missing_features)

print("\nTargets:")
print(missing_targets)


# ------------------------------------------------------------
# 7. Target statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TARGET STATISTICS")
print("=" * 70)

print(
    y.describe()
)


# ------------------------------------------------------------
# 8. G1 / G2 / G3 relationship
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("G1 / G2 / G3 CORRELATION")
print("=" * 70)

print(
    y.corr()
)


# ------------------------------------------------------------
# 9. Study time distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STUDY TIME DISTRIBUTION")
print("=" * 70)

print(
    X["studytime"].value_counts().sort_index()
)


# ------------------------------------------------------------
# 10. Absence statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ABSENCE STATISTICS")
print("=" * 70)

print(
    X["absences"].describe()
)


# ------------------------------------------------------------
# 11. G3 distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL GRADE (G3) DISTRIBUTION")
print("=" * 70)

print(
    y["G3"].value_counts().sort_index()
)


# ------------------------------------------------------------
# 12. Unique values for important variables
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("IMPORTANT FEATURE VALUES")
print("=" * 70)

important_features = [
    "studytime",
    "failures",
    "absences",
    "higher",
    "internet",
    "health"
]

for feature in important_features:

    print(
        f"\n{feature}:"
    )

    print(
        X[feature].value_counts().sort_index()
    )


# ------------------------------------------------------------
# 13. First five complete records
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FIRST 5 RECORDS")
print("=" * 70)

combined = pd.concat(
    [X, y],
    axis=1
)

print(
    combined.head().to_string()
)


# ------------------------------------------------------------
# 14. Final message
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)

print(
    "\nNo project dataset has been replaced."
)

print(
    "No existing model has been changed."
)

print(
    "\nThe results will be used to design "
    "the final prediction problem."
)

print()
import pandas as pd


print("\n" + "=" * 70)
print("CANDIDATE STUDENT EXAM DATASET INSPECTION")
print("=" * 70)


# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path = "data/raw/student_exam_scores.csv"

data = pd.read_csv(file_path)

print("\nDataset loaded successfully.")


# ============================================================
# 2. DATASET SIZE
# ============================================================

print("\n" + "=" * 70)
print("DATASET SIZE")
print("=" * 70)

print("\nNumber of rows:", data.shape[0])
print("Number of columns:", data.shape[1])


# ============================================================
# 3. COLUMN NAMES
# ============================================================

print("\n" + "=" * 70)
print("COLUMN NAMES")
print("=" * 70)

for i, column in enumerate(data.columns, start=1):
    print(f"{i:2d}. {column}")


# ============================================================
# 4. DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(data.dtypes)


# ============================================================
# 5. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = data.isnull().sum()

print(missing)

print(
    "\nTotal missing values:",
    missing.sum()
)


# ============================================================
# 6. DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE ROWS")
print("=" * 70)

duplicates = data.duplicated().sum()

print(
    "Number of duplicate rows:",
    duplicates
)


# ============================================================
# 7. FIRST 10 ROWS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 10 ROWS")
print("=" * 70)

print(
    data.head(10).to_string(index=False)
)


# ============================================================
# 8. BASIC STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("NUMERICAL STATISTICS")
print("=" * 70)

print(
    data.describe().to_string()
)


# ============================================================
# 9. UNIQUE VALUES
# ============================================================

print("\n" + "=" * 70)
print("UNIQUE VALUES")
print("=" * 70)

for column in data.columns:

    print(
        f"\n{column}:"
    )

    print(
        "Number of unique values:",
        data[column].nunique()
    )

    if data[column].nunique() <= 15:

        print(
            data[column].unique()
        )


# ============================================================
# 10. CORRELATION
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION MATRIX")
print("=" * 70)

numeric_data = data.select_dtypes(
    include="number"
)

print(
    numeric_data.corr().round(3).to_string()
)


# ============================================================
# 11. CHECK EXPECTED FEATURES
# ============================================================

print("\n" + "=" * 70)
print("EXPECTED PROJECT FEATURES")
print("=" * 70)

expected_features = [
    "hours_studied",
    "attendance_percent",
    "previous_scores",
    "exam_score"
]

for feature in expected_features:

    if feature in data.columns:

        print(
            f"✓ {feature} found"
        )

    else:

        print(
            f"✗ {feature} NOT FOUND"
        )


# ============================================================
# 12. RANGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("IMPORTANT VALUE RANGES")
print("=" * 70)

for feature in expected_features:

    if feature in data.columns:

        print(
            f"\n{feature}"
        )

        print(
            "Minimum:",
            data[feature].min()
        )

        print(
            "Maximum:",
            data[feature].max()
        )

        print(
            "Mean:",
            round(data[feature].mean(), 2)
        )


# ============================================================
# 13. TARGET DISTRIBUTION
# ============================================================

if "exam_score" in data.columns:

    print("\n" + "=" * 70)
    print("EXAM SCORE DISTRIBUTION")
    print("=" * 70)

    print(
        data["exam_score"]
        .describe()
        .to_string()
    )


# ============================================================
# 14. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)

print(
    "\nNo existing project dataset was changed."
)

print(
    "No model was trained or replaced."
)

print(
    "\nThe candidate dataset is ready for evaluation."
)

print()
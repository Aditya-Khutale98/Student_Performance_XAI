import pandas as pd
import numpy as np

# ============================================================
# LOAD DATASET
# ============================================================

DATA_PATH = "data/raw/StudentPerformanceFactors.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("STUDENT PERFORMANCE FACTORS - DETAILED ANALYSIS")
print("=" * 70)

# ============================================================
# BASIC INFORMATION
# ============================================================

print("\nDATASET SHAPE")
print("-" * 70)
print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

# ============================================================
# IMPORTANT COLUMNS
# ============================================================

features = [
    "Hours_Studied",
    "Attendance",
    "Previous_Scores"
]

target = "Exam_Score"

print("\nIMPORTANT VARIABLES")
print("-" * 70)

for col in features + [target]:
    print(f"\n{col}")
    print(f"  Minimum : {df[col].min()}")
    print(f"  Maximum : {df[col].max()}")
    print(f"  Mean    : {df[col].mean():.2f}")
    print(f"  Median  : {df[col].median():.2f}")
    print(f"  Std     : {df[col].std():.2f}")
    print(f"  Unique  : {df[col].nunique()}")

# ============================================================
# CORRELATION
# ============================================================

print("\n\nCORRELATION WITH EXAM SCORE")
print("-" * 70)

correlations = df[
    ["Hours_Studied", "Attendance", "Previous_Scores", "Exam_Score"]
].corr()["Exam_Score"].sort_values(ascending=False)

print(correlations)

# ============================================================
# MISSING VALUES
# ============================================================

print("\n\nMISSING VALUES")
print("-" * 70)

missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values.")
else:
    print(missing)

# ============================================================
# DUPLICATES
# ============================================================

print("\n\nDUPLICATE ROWS")
print("-" * 70)

print(df.duplicated().sum())

# ============================================================
# SCORE DISTRIBUTION
# ============================================================

print("\n\nEXAM SCORE DISTRIBUTION")
print("-" * 70)

print(
    df["Exam_Score"]
    .describe()
    .to_string()
)

# ============================================================
# STUDY HOURS DISTRIBUTION
# ============================================================

print("\n\nSTUDY HOURS DISTRIBUTION")
print("-" * 70)

print(
    df["Hours_Studied"]
    .value_counts()
    .sort_index()
    .to_string()
)

# ============================================================
# CONVERT WEEKLY HOURS TO DAILY HOURS
# ============================================================

df["Study_Hours_Daily_Equivalent"] = df["Hours_Studied"] / 7

print("\n\nDAILY STUDY-HOUR EQUIVALENT")
print("-" * 70)

print(
    df["Study_Hours_Daily_Equivalent"]
    .describe()
    .to_string()
)

# ============================================================
# DAILY HOUR RANGES
# ============================================================

print("\n\nDAILY STUDY HOUR RANGES")
print("-" * 70)

bins = [0, 1, 2, 3, 4, 5, 6, 7, 8]

labels = [
    "0-1",
    "1-2",
    "2-3",
    "3-4",
    "4-5",
    "5-6",
    "6-7",
    "7-8"
]

df["Daily_Hour_Group"] = pd.cut(
    df["Study_Hours_Daily_Equivalent"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

print(
    df["Daily_Hour_Group"]
    .value_counts()
    .sort_index()
    .to_string()
)

# ============================================================
# AVERAGE EXAM SCORE BY STUDY HOURS
# ============================================================

print("\n\nAVERAGE EXAM SCORE BY WEEKLY STUDY HOURS")
print("-" * 70)

study_analysis = (
    df.groupby("Hours_Studied", observed=True)["Exam_Score"]
    .agg(["count", "mean", "std"])
)

print(study_analysis.to_string())

# ============================================================
# AVERAGE EXAM SCORE BY ATTENDANCE GROUP
# ============================================================

print("\n\nAVERAGE EXAM SCORE BY ATTENDANCE")
print("-" * 70)

attendance_analysis = (
    df.groupby("Attendance")["Exam_Score"]
    .agg(["count", "mean", "std"])
)

print(attendance_analysis.to_string())

# ============================================================
# AVERAGE EXAM SCORE BY PREVIOUS SCORE
# ============================================================

print("\n\nAVERAGE EXAM SCORE BY PREVIOUS SCORE")
print("-" * 70)

previous_analysis = (
    df.groupby("Previous_Scores")["Exam_Score"]
    .agg(["count", "mean", "std"])
)

print(previous_analysis.to_string())

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)
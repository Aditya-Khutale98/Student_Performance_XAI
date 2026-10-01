import pandas as pd
import matplotlib.pyplot as plt

file_path = "data/raw/student_exam_performance.csv"

data = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print(data.head())

plt.scatter(data["hours_studied"], data["exam_score"])
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score")
plt.show()

plt.scatter(data["attendance_percentage"], data["exam_score"])
plt.xlabel("Attendance (%)")
plt.ylabel("Exam Score")
plt.title("Attendance vs Exam Score")
plt.show()

plt.scatter(data["sleep_hours"], data["exam_score"])
plt.xlabel("Sleep Hours")
plt.ylabel("Exam Score")
plt.title("Sleep Hours vs Exam Score")
plt.show()

print("\nCorrelation Matrix:")
print(
    data[
        [
            "hours_studied",
            "attendance_percentage",
            "sleep_hours",
            "exam_score"
        ]
    ].corr()
)
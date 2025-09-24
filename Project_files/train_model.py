
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import shap
import joblib

df = pd.read_csv("student_scores_dataset.csv")
df = df.drop(columns=["id", "first_name", "last_name", "email"])

le_dict = {}
for col in ['gender', 'part_time_job', 'extracurricular_activities', 'career_aspiration']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    joblib.dump(le, f"{col}_encoder.pkl")

score_cols = ['math_score', 'history_score', 'physics_score', 'chemistry_score',
              'biology_score', 'english_score', 'geography_score']

df["pass_fail"] = (df[score_cols] >= 40).sum(axis=1) >= 6  # At least 6 out of 7 subjects passed
df["pass_fail"] = df["pass_fail"].astype(int)

y = df["pass_fail"]
X = df.drop(columns=["pass_fail"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("✅ Model trained and saved successfully.")

# 📘 AI-powered EduPredict – Student Performance and Analytics System  

## 📌 Introduction  
EduPredict is an AI-powered system designed to analyze student academic and behavioral data to predict performance outcomes such as **Pass/Fail classification, dropout risk, and overall academic trends**. By leveraging **Machine Learning** models, the system provides institutions and educators with **early intervention insights**, helping improve student success rates.  

---

## 🎯 Objectives  
- Predict student performance based on academic and behavioral attributes.  
- Identify patterns and risk factors that affect academic success.  
- Provide educators with actionable insights to support struggling students.  
- Develop a user-friendly analytics dashboard for visualization.  

---

## ✨ Features  
- 📊 **Performance Prediction** – Predicts student outcomes (pass/fail or grades).  
- 🧠 **Machine Learning Models** – Uses classification algorithms like Random Forest, Logistic Regression, and XGBoost.  
- 🔍 **Analytics Dashboard** – Visualizes trends, failure risk, and subject-wise performance.  
- 📈 **SHAP-based Explainability** – Explains why a prediction was made.  
- 📂 **Custom Dataset Handling** – Works with student academic and behavioral datasets.  

---

## 📂 Dataset  
The dataset includes:  
- Student academic scores (subject-wise marks, GPA, previous performance).  
- Attendance records.  
- Behavioral attributes (participation, assignments, activities).  
- Target label: **Pass/Fail or Grade classification**.  

---

## 🛠️ Tech Stack  
- **Programming Language**: Python 🐍  
- **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn  
- **Visualization**: Streamlit (interactive dashboard)  
- **Database**: SQLite / CSV for dataset storage  
- **Version Control**: Git & GitHub  

---

## 🚀 Installation & Usage  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/Vivekchary2607/AI-Powered-EduPredict-Student-Performance-Analytics-System.git)
cd edupredict
```
### 2️⃣ Create virtual environment & install dependencies
```bash
python -m venv venv
# Activate environment
source venv/bin/activate   # for Linux/Mac
venv\Scripts\activate      # for Windows

# Install requirements
pip install -r requirements.txt
```
### 3️⃣ Run the Streamlit app
```bash
streamlit run app.py
```
### 4️⃣ Upload dataset & start predictions 🎓
---
### 📊 Results

- Achieved 96% accuracy using Random Forest.

- SHAP values highlighted key factors: subject performance, attendance, and assignment completion.

- Provided interpretable insights for educators to take timely action.

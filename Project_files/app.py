# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import json
from database import validate_user, add_user, save_prediction, get_all_predictions

# ------------------- Utilities -------------------
def safe_encode_series(series, encoder):
    classes = list(encoder.classes_)
    mapping = {v: i for i, v in enumerate(classes)}
    default_idx = 0
    return series.map(mapping).fillna(default_idx).astype(int)

def generate_explanation(pred_prob, prediction, input_data):
    reasons_positive, reasons_negative = [], []

    if pred_prob > 0.85 or pred_prob < 0.15:
        certainty = "with high certainty"
    elif 0.65 <= pred_prob <= 0.85 or 0.15 <= pred_prob <= 0.35:
        certainty = "with moderate certainty"
    else:
        certainty = "with low certainty"

    if input_data['math_score'].values[0] < 40:
        reasons_negative.append(f"low math score ({int(input_data['math_score'].values[0])})")
    elif input_data['math_score'].values[0] > 70:
        reasons_positive.append(f"strong math score ({int(input_data['math_score'].values[0])})")

    if input_data['english_score'].values[0] < 40:
        reasons_negative.append(f"low English score ({int(input_data['english_score'].values[0])})")
    elif input_data['english_score'].values[0] > 70:
        reasons_positive.append(f"strong English score ({int(input_data['english_score'].values[0])})")

    if input_data['weekly_self_study_hours'].values[0] == 0:
        reasons_negative.append("no self-study hours")
    elif input_data['weekly_self_study_hours'].values[0] > 5:
        reasons_positive.append("consistent self-study")

    if input_data['extracurricular_activities'].values[0] == 0:
        reasons_negative.append("no extracurricular activities")
    else:
        reasons_positive.append("involvement in extracurriculars")

    if input_data['absence_days'].values[0] > 20:
        reasons_negative.append(f"{int(input_data['absence_days'].values[0])} absence days")
    elif input_data['absence_days'].values[0] < 5:
        reasons_positive.append("very few absences")

    if prediction == 1:
        msg = f"This student is predicted to **Pass** {certainty} (model output: {pred_prob:.2f}). "
        if reasons_positive:
            msg += "Supported by " + ", ".join(reasons_positive) + ". "
        if reasons_negative:
            msg += "Some risks: " + ", ".join(reasons_negative) + "."
    else:
        msg = f"This student is predicted to **Fail** {certainty} (model output: {pred_prob:.2f}). "
        if reasons_negative:
            msg += "Driven by " + ", ".join(reasons_negative) + ". "
        if reasons_positive:
            msg += "Positive signals: " + ", ".join(reasons_positive) + "."

    return msg

# ------------------- Load Model -------------------
try:
    model = joblib.load("model.pkl")
    columns = joblib.load("model_columns.pkl")
except Exception as e:
    st.error(f"❌ Could not load model or columns. Error: {e}")
    st.stop()

# Load encoders
encoder_names = {
    "gender": "gender_encoder.pkl",
    "part_time_job": "part_time_job_encoder.pkl",
    "extracurricular_activities": "extracurricular_activities_encoder.pkl",
    "career_aspiration": "career_aspiration_encoder.pkl"
}
encoders = {}
for k, fname in encoder_names.items():
    try:
        encoders[k] = joblib.load(fname)
    except:
        encoders[k] = None

# ------------------- Streamlit Config -------------------
st.set_page_config(page_title="Student Performance System", layout="wide")
st.sidebar.markdown("## 📘 Student Performance & Analytics System")
st.sidebar.info("An AI-powered early-warning system for teachers 🚀")

# ------------------- Session State -------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None

# ------------------- Auth -------------------
if not st.session_state.logged_in:
    st.title("🔐 Authentication Required")
    choice = st.radio("Select Option:", ["Login", "Register"])

    if choice == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            role = validate_user(username, password)
            if role:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.role = role
                st.success(f"Welcome, {username} ({role})")
                st.experimental_rerun()
            else:
                st.error("Invalid credentials")

    elif choice == "Register":
        st.subheader("📝 Register New Account")
        new_username = st.text_input("Choose Username")
        new_password = st.text_input("Choose Password", type="password")
        role = st.selectbox("Role", ["teacher", "admin"])
        subject = "None"
        if role=="teacher":
            subject=st.selectbox("Subject", 
                               ["Math", "English", "Science", "History", "Biology", "Geography", "General"])

        if st.button("Register"):
            add_user(new_username, new_password, role=role ,subject=subject)
            st.success("✅ Account created! Please login now.")
    st.stop()

# ------------------- Navigation -------------------
st.sidebar.write(f"👤 Logged in as: **{st.session_state.username}** ({st.session_state.role})")
page = st.sidebar.radio("Navigation", ["Single Prediction", "Batch Prediction", "Admin Panel", "Logout"])

if page == "Logout":
    st.session_state.logged_in = False
    st.experimental_rerun()

# ------------------- Single Prediction -------------------
if page == "Single Prediction":
    st.header("🔎 Single Student Prediction")

    student_name = st.text_input("Student name (optional)")

    with st.expander("📋 Student Info"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["male", "female"])
            part_time_job = st.selectbox("Part-Time Job", ["True", "False"])
            extracurricular = st.selectbox("Extracurricular Activities", ["True", "False"])
            career = st.selectbox("Career Aspiration", 
                                  list(encoders['career_aspiration'].classes_) if encoders['career_aspiration'] else ['Unknown'])
        with col2:
            absence_days = st.number_input("Absence Days", 0, 365, step=1)
            weekly_study_hours = st.slider("Weekly Self Study Hours", 0, 40)
            math = st.slider("Math Score", 0, 100)
            history = st.slider("History Score", 0, 100)
            physics = st.slider("Physics Score", 0, 100)
            chemistry = st.slider("Chemistry Score", 0, 100)
            biology = st.slider("Biology Score", 0, 100)
            english = st.slider("English Score", 0, 100)
            geography = st.slider("Geography Score", 0, 100)

    if st.button("Predict"):
        input_data = pd.DataFrame([[gender, part_time_job, extracurricular, career,
                                    absence_days, weekly_study_hours,
                                    math, history, physics, chemistry, biology, english, geography]],
                                  columns=["gender", "part_time_job", "extracurricular_activities", "career_aspiration",
                                           "absence_days", "weekly_self_study_hours",
                                           "math_score", "history_score", "physics_score", "chemistry_score",
                                           "biology_score", "english_score", "geography_score"])

        # encode categorical safely
        for col in ["gender", "part_time_job", "extracurricular_activities", "career_aspiration"]:
            enc = encoders.get(col)
            if enc:
                input_data[col] = safe_encode_series(input_data[col], enc)

        for c in columns:
            if c not in input_data.columns:
                input_data[c] = 0
        input_data = input_data[columns]

        prediction = model.predict(input_data)[0]
        pred_prob = model.predict_proba(input_data)[0][1]

        # Result Card
        if prediction == 1:
            st.success(f"🎯 Prediction: PASS ✅ (Probability: {pred_prob:.2f})")
        else:
            st.error(f"🎯 Prediction: FAIL ❌ (Probability: {pred_prob:.2f})")

        # Bar chart
        scores = {"Math": math, "History": history, "Physics": physics, "Chemistry": chemistry,
                "Biology": biology, "English": english, "Geography": geography}
        colors = ['green' if v >= 40 else 'red' for v in scores.values()]

        fig, ax = plt.subplots(figsize=(3.5, 2.2))  # smaller size
        ax.bar(scores.keys(), scores.values(), color=colors)
        ax.axhline(y=40, color='black', linestyle='--')
        ax.set_ylabel("Marks", fontsize=8)
        ax.set_title("Subject-wise Scores", fontsize=10)
        ax.tick_params(axis='x', labelrotation=30, labelsize=8)  
        ax.tick_params(axis='y', labelsize=8)
        st.pyplot(fig, clear_figure=True,use_container_width=False)

        # SHAP Explanation
        explainer = shap.Explainer(model, input_data)
        shap_values = explainer(input_data)

        st.markdown("### 🔍 SHAP Decision Plot for Prediction")
        st.write("Shows how each feature moves the prediction score towards Pass or Fail.")

        plt.rcParams.update({'font.size': 0.3})  
        fig= plt.figure(figsize=(3.5, 2.5))  
        shap.decision_plot(
            base_value=explainer.expected_value[1],
            shap_values=shap_values.values[0, :, 1],
            features=input_data,
            feature_names=list(input_data.columns),
            show=False
        )
        st.pyplot(fig, clear_figure=True)


        # Text Explanation
        explanation = generate_explanation(pred_prob, prediction, input_data)
        st.markdown("### 📌 Prediction Explanation")
        st.markdown(explanation)

        # Save to DB
        student_display_name = student_name if student_name else "Unnamed Student"
        save_prediction(st.session_state.username, student_display_name,
                        json.loads(input_data.to_json(orient="records"))[0],
                        "Pass" if prediction == 1 else "Fail",
                        float(pred_prob), explanation)
        st.success("Prediction saved to database.")

# ------------------- Batch Prediction -------------------
elif page == "Batch Prediction":
    st.header("📂 Batch Prediction (Upload CSV)")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        for col in ["gender", "part_time_job", "extracurricular_activities", "career_aspiration"]:
            enc = encoders.get(col)
            if enc is not None:
                df[col] = safe_encode_series(df[col], enc)

        for c in columns:
            if c not in df:
                df[c] = 0
        df_model = df[columns]

        preds = model.predict(df_model)
        probs = model.predict_proba(df_model)[:, 1]
        df["Prediction"] = ["Pass" if p == 1 else "Fail" for p in preds]
        df["Probability"] = probs.round(3)

        explanations = []
        for i, row in df_model.iterrows():
            one = pd.DataFrame([row.values], columns=df_model.columns)
            explanations.append(generate_explanation(probs[i], preds[i], one))
        df["Explanation"] = explanations

        st.dataframe(df.head(200))
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download predictions CSV", csv_data, "predictions.csv")

        # Save batch predictions
        for i, row in df.iterrows():
            student_name = row.get("student_name", f"Student_{i+1}")
            save_prediction(st.session_state.username,
                            student_name,
                            json.loads(df_model.iloc[i:i+1].to_json(orient="records"))[0],
                            row["Prediction"],
                            float(row["Probability"]),
                            row["Explanation"])
        st.success("All batch predictions saved to database.")

# ------------------- Admin Panel -------------------
elif page == "Admin Panel":
    if st.session_state.role["role"] != "admin":
        st.error("Admin access only")
        st.stop()
    # ---------- Admin Panel ---------
    st.header("🔐 Admin Panel")

    # Tabs for admin functions
    tab1, tab2, tab3 = st.tabs(["📊 Predictions", "👤 Manage Users", "⚠️ Database Tools"])

    # --- Predictions Tab ---
    with tab1:
        st.subheader("All Saved Predictions")
        all_preds = get_all_predictions()
        if not all_preds:
            st.info("No predictions saved yet.")
        else:
            df_all = pd.DataFrame(all_preds)

            #Filtering options
            teacher_filter = st.selectbox("Filter by Teacher", ["All"] + df_all["username"].unique().tolist())
            student_filter = st.text_input("Filter by Student Name")

            df_filtered = df_all.copy()
            if teacher_filter != "All":
                df_filtered = df_filtered[df_filtered["username"] == teacher_filter]
            if student_filter:
                df_filtered = df_filtered[df_filtered["student_name"].str.contains(student_filter, case=False, na=False)]

            st.dataframe(df_filtered, use_container_width=True)

            # Option to download
            csv_all = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button("Download filtered predictions CSV", csv_all, "filtered_predictions.csv")

    # --- Manage Users Tab ---
    with tab2:
        st.subheader("User Management")

        from database import get_all_users, delete_user, add_user

        users = get_all_users()
        if users:
            df_users = pd.DataFrame(users)
            st.dataframe(df_users, use_container_width=True)

            del_user = st.selectbox("Select a user to delete", ["None"] + df_users["username"].tolist())
            if del_user != "None" and st.button("Delete User"):
                delete_user(del_user)
                st.success(f"✅ User '{del_user}' deleted.")
                st.experimental_rerun()
        else:
            st.info("No users found.")

        st.markdown("### ➕ Add New User")
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        role = st.selectbox("Role", ["teacher", "admin"])

        subject = None
        if role == "teacher":
            subject = st.selectbox("Subject", ["Math", "Science", "English", "History", "Biology", "General"])

        if st.button("Add User"):
            if new_username and new_password:
                if role == "teacher":
                    add_user(new_username, new_password, role="teacher", subject=subject)
                else:
                    add_user(new_username, new_password, role="admin")
                st.success(f"✅ User '{new_username}' added successfully!")
                st.experimental_rerun()
            else:
                st.error("Username and password required.")

    #  Database Tools Tab ---
    with tab3:
        st.subheader("⚠️ Database Tools")
        st.warning("Be careful! These actions are irreversible.")

        if st.button("🗑️ Clear All Predictions"):
            from database import clear_predictions
            clear_predictions()
            st.success("✅ All predictions deleted successfully.")
            st.experimental_rerun()

        if st.button("🗑️ Clear All Users (except admin)"):
            from database import clear_non_admins
            clear_non_admins()
            st.success("✅ All non-admin users deleted successfully.")
            st.experimental_rerun()

import streamlit as st
import pickle
import pandas as pd

# ─── 0) PAGE CONFIG MUST BE FIRST ─────────────────────────────
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💼",
    layout="centered"
)

# ─── 1) LOAD MODEL ────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("Final Pickle.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ─── 2) APP HEADER ────────────────────────────────────────────
st.title("💼 Data Scientist Salary Predictor")
st.subheader("📈 Estimate your annual salary based on your profile")

# ─── 3) USER INPUTS ───────────────────────────────────────────
education_mapping = {
    "High School or Some College": 0,
    "Bachelor’s Degree":          1,
    "Master’s Degree":            2,
    "Doctoral Degree":            3
}

education = st.selectbox(
    "Highest Formal Education",
    list(education_mapping.keys()),
)

years_coding = st.slider(
    "Years of Coding Experience",
    min_value=0, max_value=40, value=5, step=1
)

country = st.selectbox(
    "Country of Residence",
    ["Canada","India","US","Spain","Other"]
)

codes_java   = st.checkbox("I code in Java")
codes_python = st.checkbox("I code in Python")
codes_sql    = st.checkbox("I code in SQL")
codes_go     = st.checkbox("I code in Go")

# ─── 4) BUILD FEATURE VECTOR ───────────────────────────────────
features = {
    "Education":       education_mapping[education],
    "Years_Coding":    years_coding,
    "Codes_In_JAVA":   int(codes_java),
    "Codes_In_Python": int(codes_python),
    "Codes_In_SQL":    int(codes_sql),
    "Codes_In_GO":     int(codes_go),
    "Country_India":   0,
    "Country_Other":   0,
    "Country_Spain":   0,
    "Country_US":      0,
}

if country != "Canada":
    features[f"Country_{country}"] = 1

X = pd.DataFrame([features])

# ─── 5) PREDICTION ─────────────────────────────────────────────
st.markdown("---")
st.subheader("💵 Your Predicted Salary")

if st.button("Predict"):
    salary_est = model.predict(X)[0]
    st.success(f"Estimated Annual Salary: **${salary_est:,.2f}**")

st.markdown("---")
st.markdown(
    "<small>🔍 Model: Lasso Regression | Data: 2022 Kaggle DS Survey</small>",
    unsafe_allow_html=True
)

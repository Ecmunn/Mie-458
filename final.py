import streamlit as st
import pickle
import pandas as pd

# ─── 1) MODEL LOADING ────────────────────────────────────────

@st.cache_resource
def load_model():
    with open("Final Pickle.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()


# ─── 2) UI & INPUTS ─────────────────────────────────────────

st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("💼 Data Scientist Salary Predictor")

# Education mapping must match what you used in training
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

# Years coding slider (we’ll just let them pick a number 0–40)
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


# ─── 3) PREPARE FEATURE VECTOR ──────────────────────────────

# Base features
features = {
    "Education":       education_mapping[education],
    "Years_Coding":    years_coding,
    "Codes_In_JAVA":   int(codes_java),
    "Codes_In_Python": int(codes_python),
    "Codes_In_SQL":    int(codes_sql),
    "Codes_In_GO":     int(codes_go),
    # one‐hot dummies, Canada is the reference => all zeros if Canada
    "Country_India":  0,
    "Country_Other":  0,
    "Country_Spain":  0,
    "Country_US":     0,
}

# turn on the correct country dummy
if country != "Canada":
    features[f"Country_{country}"] = 1

X = pd.DataFrame([features])


# ─── 4) MAKE PREDICTION ──────────────────────────────────────

st.markdown("---")
st.subheader("Estimate your salary")

if st.button("💵 Predict Salary"):
    pred = model.predict(X)[0]
    st.success(f"**Estimated Annual Salary:** ${pred:,.2f}")

st.markdown("---")
st.markdown(
    "<small>🔍 Model: Lasso Regression &#124; Data from 2022 Kaggle DS Survey</small>",
    unsafe_allow_html=True
)

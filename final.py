import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ─── 0) PAGE CONFIG ─────────────────────────────────────────
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💼",
    layout="centered"
)

# ─── 1) LOAD MODEL ───────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("Final Pickle.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ─── 2) APP HEADER ───────────────────────────────────────────
st.title("💼 Data Scientist Salary Predictor")
st.subheader("📈 Estimate your annual salary based on your profile")

# ─── 3) USER INPUTS ──────────────────────────────────────────
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
education_num = education_mapping[education]

years_coding = st.slider(
    "Years of Coding Experience",
    min_value=0, max_value=40, value=5, step=1
)

country = st.selectbox(
    "Country of Residence",
    ["Canada", "India", "US", "Spain", "Other"]
)

codes_java   = st.checkbox("I code in Java")
codes_python = st.checkbox("I code in Python")
codes_sql    = st.checkbox("I code in SQL")
codes_go     = st.checkbox("I code in Go")

# ─── 4) BUILD INPUT DATAFRAME ────────────────────────────────
# Get the exact feature list the model expects
feature_names = list(model.feature_names_in_)

# Create one blank row with all zeros
input_df = pd.DataFrame(
    np.zeros((1, len(feature_names))),
    columns=feature_names
)

# Populate numeric and binary features
input_df.at[0, "Education"]     = education_num
input_df.at[0, "Years_Coding"]  = years_coding
input_df.at[0, "Codes_In_JAVA"]   = int(codes_java)
input_df.at[0, "Codes_In_Python"] = int(codes_python)
input_df.at[0, "Codes_In_SQL"]    = int(codes_sql)
input_df.at[0, "Codes_In_GO"]     = int(codes_go)

# Set the correct country dummy (Canada is reference, so leave all zeros if Canada)
col = f"Country_{country}"
if col in input_df.columns:
    input_df.at[0, col] = 1

# ─── 5) PREDICTION ───────────────────────────────────────────
st.markdown("---")
st.subheader("💵 Your Predicted Salary")

if st.button("Predict Salary"):
    salary_est = model.predict(input_df)[0]
    st.success(f"Estimated Annual Salary: **${salary_est:,.2f}**")

st.markdown("---")
st.markdown(
    "<small>🔍 Model: Lasso Regression | Data: 2022 Kaggle DS Survey</small>",
    unsafe_allow_html=True
)


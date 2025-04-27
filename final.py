# final.py

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ─────────────────────────────────────────────────────────────────────────────
# 1) LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(
    'kaggle_survey_2022_responses.csv',  
    skiprows=1,
    low_memory=False
)
df.columns = df.columns.str.strip()

# select + rename
df = df[[
    'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?',
    'For how many years have you been writing code and/or programming?',
    'In which country do you currently reside?',
    'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Java',
    'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Python',
    'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - SQL',
    'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Go',
    'What is your current yearly compensation (approximate $USD)?'
]].copy()

df.columns = [
    'Education', 'Years_Coding', 'Country',
    'Codes_In_Java', 'Codes_In_Python', 'Codes_In_SQL', 'Codes_In_GO',
    'Salary'
]

# map education
education_map = {
    "No formal education past high school": 0,
    "Some college/university study without earning a bachelor’s degree": 0,
    "Bachelor’s degree": 1,
    "Master’s degree": 2,
    "Doctoral degree": 3,
    "Professional doctorate": 3,
    "I prefer not to answer": np.nan
}
df['Education'] = df['Education'].map(education_map)

# binary encode languages
for c in ['Codes_In_Java','Codes_In_Python','Codes_In_SQL','Codes_In_GO']:
    df[c] = df[c].notna().astype(int)

# salary → midpoint
def clean_salary(s):
    if isinstance(s, str):
        s2 = s.replace('$','').replace(',','')
        if '-' in s2:
            lo, hi = s2.split('-')
            return (float(lo)+float(hi))/2
        if s2.isdigit():
            return float(s2)
    return np.nan

df['Salary'] = df['Salary'].apply(clean_salary)

# drop any rows with missing
df = df.dropna()

# ─────────────────────────────────────────────────────────────────────────────
# 2) FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
# clean Years_Coding
def clean_years(x):
    if isinstance(x, str):
        if '10-20' in x:      return 15
        if '20+' in x:        return 25
        if '5-10' in x:       return 7.5
        if '3-5' in x:        return 4
        if '1-3' in x:        return 2
        if '< 1' in x:        return 0.5
        if 'never written' in x: return 0
    return x

df['Years_Coding'] = df['Years_Coding'].apply(clean_years)

# one-hot encode Country
df = pd.get_dummies(df, columns=['Country'], drop_first=True)

# split X / y
X = df.drop(columns='Salary')
y = df['Salary']

# train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────────────────────────────────────
# 3) MODELING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
# -- Linear Regression
linreg = LinearRegression().fit(X_train, y_train)
lin_pred = linreg.predict(X_test)

# -- Decision Tree
tree = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
tree_pred = tree.predict(X_test)

# -- Lasso
lasso = Lasso(alpha=0.1, random_state=42).fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)

# metrics
def report(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"{name:<15} RMSE: {rmse:,.2f}    R²: {r2:.3f}")

report("LinearReg", y_test, lin_pred)
report("DecisionTree", y_test, tree_pred)
report("Lasso",      y_test, lasso_pred)

# correlation matrix
comparison = pd.DataFrame({
    'Actual':         y_test,
    'Linear_Pred':    lin_pred,
    'Tree_Pred':      tree_pred,
    'Lasso_Pred':     lasso_pred
})
corr = comparison.corr()

plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis")
plt.title("Actual vs Predicted Salary Correlations")
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 4) PICKLE (just Lasso as final model)
# ─────────────────────────────────────────────────────────────────────────────
with open('lasso_salary_model.pkl','wb') as f:
    pickle.dump(lasso, f)

print("✅ Model pickled to lasso_salary_model.pkl")


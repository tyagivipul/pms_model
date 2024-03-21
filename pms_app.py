import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('hcm_pms_test.csv')

# Preprocessing
df['DOJ'] = pd.to_datetime(df['DOJ'], errors='coerce')
current_date = datetime.now()
df['TENURE_YEARS'] = df['DOJ'].apply(lambda x: (current_date - x).days / 365 if pd.notnull(x) else None)
df['ATTRITION'] = df['LWD'].apply(lambda x: 0 if pd.isnull(x) else 1)

# Feature Selection
features = ['TENURE_YEARS', 'PMS_SCORE', 'ATTD_SCORE', 'ACHIEVMENT_SCORE', 'DISPLAY_SCORE', 'TRAINING_SCORE', 'PRESENT_CNT', 'ABSENT_CNT', 'LEAVE_CNT', 'WEEKOFF_CNT', 'ATTRITION']
df_model = df[features]

# Separating the features and the target variable
X = df_model.drop('ATTRITION', axis=1)
y = df_model['ATTRITION']

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Streamlit UI
st.title('Employee Attrition Prediction')

with st.form("attrition_form"):
    st.write("Please provide the following employee details:")
    tenure_years = st.number_input('Tenure in years:', min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    pms_score = st.number_input('PMS score (out of 100):', min_value=0.0, max_value=100.0, value=80.0, step=1.0)
    attd_score = st.number_input('Attendance score (out of 25):', min_value=0.0, max_value=25.0, value=20.0, step=1.0)
    achievment_score = st.number_input('Achievement score:', min_value=0.0, max_value=100.0, value=75.0, step=1.0)
    display_score = st.number_input('Display score (out of 15):', min_value=0.0, max_value=15.0, value=10.0, step=1.0)
    training_score = st.number_input('Training score (out of 15):', min_value=0.0, max_value=15.0, value=10.0, step=1.0)
    present_cnt = st.number_input('Present count (out of 30 days):', min_value=0.0, max_value=30.0, value=25.0, step=1.0)
    absent_cnt = st.number_input('Absent count(out of 30 days):', min_value=0.0, max_value=30.0, value=2.0, step=1.0)
    leave_cnt = st.number_input('Leave count (out of 30 days):', min_value=0.0, max_value=30.0, value=2.0, step=1.0)
    weekoff_cnt = st.number_input('Week off count(out of 30 days):', min_value=0.0, max_value=30.0, value=4.0, step=1.0)

    submitted = st.form_submit_button("Predict Attrition")
    if submitted:
        employee_features = np.array([tenure_years, pms_score, attd_score, achievment_score, display_score, training_score, present_cnt, absent_cnt, leave_cnt, weekoff_cnt]).reshape(1, -1)
        prediction = model.predict(scaler.transform(employee_features))
        st.write("The employee is predicted to be:", "Attrited" if prediction == 1 else "Active")
import streamlit as st
import torch
import numpy as np
import pandas as pd
import joblib
from MLP import MLP
import xgboost as xgb



st.title("School-Business Revenue Predictor")

school_level = st.selectbox("Select School Level", ["High", "Middle"])
school_type = st.selectbox("Select School Type", ["Public", "Private"])
enrollment_amount = st.number_input("Number of Enrolled Students", min_value=1, value=600)
teacher_amount = st.number_input("Number of Teaching Staff", min_value=1, value=50)
average_income = st.number_input("Average Income for Family in Area", min_value=0, value=60000)
business_type = st.selectbox("Select Business Type", [
    "Club Fundraiser", "School Store & Snack Shop", "School Store",
    "Concessions", "Culinary Shop", "Plant & Flower Fundraiser",
    "Prom & Homecoming tickets"
])
maximum_operating_time = st.number_input("Expected Operating Time (days)", min_value=1, value=180)
predictive_model = "XGBoost"

#if st.button("Predict"):
if predictive_model == "XGBoost":

    encoder = joblib.load("XGB_model/encoder.pkl")
    rev_min, rev_max = joblib.load("XGB_model/rev_min_max.pkl")
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model("XGB_model/model.json")
    

    rev_predictions = {"time":[],"rev":[]}
    for operating_time in range(1, maximum_operating_time):
        # Encode categorical inputs
        cat_input = encoder.transform([[school_level.lower(), business_type.lower(), school_type.lower()]])
        num_input = np.array([[operating_time, teacher_amount, enrollment_amount, average_income]])
        x_input = np.hstack([cat_input, num_input])
        x_tensor = torch.tensor(x_input, dtype=torch.float32)

        # Run model
        pred = (xgb_model.predict(x_tensor)[0]) * (rev_max - rev_min) + rev_min
        rev_predictions['time'].append(operating_time)
        rev_predictions['rev'].append(pred)

    st.area_chart(rev_predictions, x="time", y="rev", x_label="Operating Time (Days)", y_label="Revenue ($)")

    #st.write(f"**Predicted Overall Revenue ($):** {rev_pred * operating_time:,.2f}")
    #st.write(f"**Predicted Daily Revenue ($):** {rev_pred:,.2f}")

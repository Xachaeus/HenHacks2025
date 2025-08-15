import streamlit as st
import torch
import numpy as np
import pandas as pd
import joblib
from MLP import MLP
import xgboost as xgb



st.title("School-Business Revenue Predictor")

school_level = st.selectbox("Select School Type", ["High", "Middle"])
business_type = st.selectbox("Select Business Type", [
    "Club Fundraiser", "School Store & Snack Shop", "School Store",
    "Concessions", "Culinary Shop", "Plant & Flower Fundraiser",
    "Prom & Homecoming tickets"
])
operating_time = st.number_input("Expected Operating Time (days)", min_value=1, value=180)
predictive_model = st.selectbox("Select Predictive Model", ["MLP", "Linear Regression", "XGBoost"])

if st.button("Predict"):
    if predictive_model == "MLP":
        # Load saved encoder and model
        encoder = joblib.load("./MLP_model/encoder.pkl")

        # Input dimension from encoder + numeric feature
        input_dim = encoder.transform([["middle", "club fundraiser"]]).shape[1] + 1
        model = MLP(input_dim=input_dim)
        model.load_state_dict(torch.load("./MLP_model/model.pth"))
        model.eval()

        # Load revenue scaling
        rev_min, rev_max = joblib.load("./MLP_model/rev_min_max.pkl")
        daily_rev_min, daily_rev_max = joblib.load("./MLP_model/rev_min_max.pkl")
        
        # Encode categorical inputs
        cat_input = encoder.transform([[school_level.lower(), business_type.lower()]])
        num_input = np.array([[operating_time]])
        x_input = np.hstack([cat_input, num_input])
        x_tensor = torch.tensor(x_input, dtype=torch.float32)

        # Run model
        with torch.no_grad():
            rev_pred = model(x_tensor)
            rev_rescaled = rev_pred.item() * (rev_max - rev_min) + rev_min

        st.write(f"**Predicted Overall Revenue ($):** {rev_rescaled * operating_time:,.2f}")
        st.write(f"**Predicted Daily Revenue ($):** {rev_rescaled:,.2f}")
        # st.write(f"**Survival ≥1 month:** {surv_pred[0,0].item()*100:.1f}%")
        # st.write(f"**Survival ≥3 months:** {surv_pred[0,1].item()*100:.1f}%")
        # st.write(f"**Survival ≥1 year:** {surv_pred[0,2].item()*100:.1f}%")'
        
    if predictive_model == "Linear Regression":
        model = joblib.load("LR_model/school_revenue_model.pkl")
        
        df = pd.DataFrame(
            [[school_level.lower(), business_type.lower()]],
            columns=["school_level", "business_type"]
        )
        
        print(df.shape)
        daily_rev = model.predict(df)[0]
        
        st.write(f"**Predicted Overall Revenue ($):** {daily_rev * operating_time:,.2f}")
        st.write(f"**Predicted Daily Revenue ($):** {daily_rev:,.2f}")
        
    if predictive_model == "XGBoost":
        encoder = joblib.load("XGB_model/encoder.pkl")
        rev_min, rev_max = joblib.load("XGB_model/rev_min_max.pkl")
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model("XGB_model/model.json")
        
        cat_input = encoder.transform([[school_level.lower(), business_type.lower()]])
        num_input = np.array([[operating_time]])
        x_input = np.hstack([cat_input, num_input])
        x_tensor = torch.tensor(x_input, dtype=torch.float32)

        # Predict
        rev_pred_norm = xgb_model.predict(x_tensor)[0]
        rev_pred = rev_pred_norm * (rev_max - rev_min) + rev_min

        st.write(f"**Predicted Overall Revenue ($):** {rev_pred * operating_time:,.2f}")
        st.write(f"**Predicted Daily Revenue ($):** {rev_pred:,.2f}")

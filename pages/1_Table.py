# pages/02_Bulk_Predictions.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import xgboost as xgb
from MLP import MLP

st.title("Bulk Predictions — All Combinations & Models")

school_levels = ["middle", "high"]
business_types = [
    "Club Fundraiser", "School Store & Snack Shop", "Concessions",
    "School Store", "Culinary Shop", "Plant & Flower Fundraiser",
    "Prom & Homecoming tickets"
]
durations = [7, 30, 365]

# ---- Load models/artifacts ----
# Linear Regression (predicts ANNUAL revenue)
lr = joblib.load("LR_model/school_revenue_model.pkl")

# XGBoost (predicts NORMALIZED ANNUAL revenue)
xgb_encoder = joblib.load("XGB_model/encoder.pkl")
rev_min_xgb, rev_max_xgb = joblib.load("XGB_model/rev_min_max.pkl")
xgb_model = xgb.XGBRegressor()
xgb_model.load_model("XGB_model/model.json")

# MLP (predicts NORMALIZED ANNUAL revenue)
mlp_encoder = joblib.load("MLP_model/encoder.pkl")
rev_min_mlp, rev_max_mlp = joblib.load("MLP_model/rev_min_max.pkl")
input_dim = mlp_encoder.transform([["middle", "club fundraiser"]]).shape[1] + 1
mlp = MLP(input_dim=input_dim)
mlp.load_state_dict(torch.load("MLP_model/model.pth", map_location="cpu"))
mlp.eval()

rows = []
for sl in school_levels:
    for bt in business_types:
        for d in durations:
            sl_l, bt_l = sl.lower(), bt.lower()

            # ---- Linear Regression ----
            df_lr = pd.DataFrame([[sl_l, bt_l, d]], columns=["school_level", "business_type", "operating_time"])
            daily_lr = float(lr.predict(df_lr)[0])
            annual_lr = daily_lr *365
            total_lr = daily_lr * d
            rows.append({
                "model": "Linear Regression",
                "school_level": sl,
                "business_type": bt,
                "operating_time": d,
                "daily_revenue": daily_lr,
                "total_revenue_for_duration": total_lr,
                "annual_revenue": annual_lr,
            })

            # ---- XGBoost ----
            cat_xgb = xgb_encoder.transform([[sl_l, bt_l]])
            feat_xgb = np.hstack([cat_xgb, [[float(d)]]])
            pred_norm_xgb = float(xgb_model.predict(feat_xgb)[0])
            daily_xgb = pred_norm_xgb * (rev_max_xgb - rev_min_xgb) + rev_min_xgb
            annual_xgb = daily_xgb * 365.0
            total_xgb = daily_xgb * d
            rows.append({
                "model": "XGBoost",
                "school_level": sl,
                "business_type": bt,
                "operating_time": d,
                "daily_revenue": daily_xgb,
                "total_revenue_for_duration": total_xgb,
                "annual_revenue": annual_xgb,
            })

            # ---- MLP ----
            cat_mlp = mlp_encoder.transform([[sl_l, bt_l]])
            feat_mlp = np.hstack([cat_mlp, [[float(d)]]])
            x_tensor = torch.tensor(feat_mlp, dtype=torch.float32)
            with torch.no_grad():
                pred_norm_mlp = float(mlp(x_tensor).item())
            daily_mlp = pred_norm_mlp * (rev_max_mlp - rev_min_mlp) + rev_min_mlp
            annual_mlp = daily_mlp * 365.0
            total_mlp = daily_mlp * d
            rows.append({
                "model": "MLP",
                "school_level": sl,
                "business_type": bt,
                "operating_time": d,
                "daily_revenue": daily_mlp,
                "total_revenue_for_duration": total_mlp,
                "annual_revenue": annual_mlp,
            })

df = pd.DataFrame(rows).sort_values(
    ["model", "school_level", "business_type", "operating_time"]
).reset_index(drop=True)

st.dataframe(
    df.style.format({
        "daily_revenue": "${:,.2f}",
        "total_revenue_for_duration": "${:,.2f}",
        "annual_revenue": "${:,.2f}",
    }),
    use_container_width=True,
)

csv = df.to_csv(index=False)
st.download_button("Download CSV", data=csv, file_name="bulk_predictions.csv", mime="text/csv")

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from helpers import load_preprocessed_data, metrics, denorm

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxmltools
import onnx


"""
Best model is trained on weekly intervals with a test_size of 0.1 ###
Generalizes to ~40% of test data, not great 
"""

# ---------------- Load dataset ----------------
# df = load_data("raw_dataset.json")
df = load_preprocessed_data("JSONs\\labeled_instantiated_dataset_8.json")

# ---------------- Features ----------------
X_cat = df[["school_level", "business_type", "school_type"]].values
X_num = df[["operating_time", "num_teachers", "num_students", "average_income"]].values.astype(float)
y_revenue = df["daily_revenue"].values.astype(float).reshape(-1,1)

# Encode categorical features
encoder = OneHotEncoder(sparse_output=False)
X_cat_enc = encoder.fit_transform(X_cat)
X = np.hstack([X_cat_enc, X_num])

# Normalize revenue (optional for XGBoost)
rev_min, rev_max = y_revenue.min(), y_revenue.max()
y_revenue_norm = (y_revenue - rev_min) / (rev_max - rev_min)

# Save encoder and min/max for inference
joblib.dump(encoder, "XGB_model\\encoder.pkl")
joblib.dump((rev_min, rev_max), "XGB_model\\rev_min_max.pkl")

# ---------------- Train/Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_revenue_norm, test_size=0.2, random_state=42
)

# ---------------- Train XGBoost ----------------
xgb_model = xgb.XGBRegressor(
    n_estimators=10000,
    learning_rate=0.001,
    max_depth=20,
    subsample=0.5,
    colsample_bytree=0.5,
    random_state=42
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=50
)

print("Train R²:", xgb_model.score(X_train, y_train))
print("Test R²:", xgb_model.score(X_test, y_test))

# ---------------- Generate predictions table ----------------
# rows = []
# avg_time = df["operating_time"].mean()  # average operating time for predictions

# for school in df["school_level"].unique():
#     for btype in df["business_type"].unique():
#         # Encode categorical features
#         ex_cat = encoder.transform([[school, btype]])
#         ex_feat = np.hstack([ex_cat, [[avg_time]]])

#         # Predict annual revenue
#         rev_pred = xgb_model.predict(ex_feat)[0]
#         rev_pred_rescaled = rev_pred * (rev_max - rev_min) + rev_min
#         daily_pred = rev_pred_rescaled / 365

#         rows.append({
#             "School": school,
#             "Business Type": btype,
#             "Predicted Annual Revenue ($)": rev_pred_rescaled * 365,
#             "Predicted Daily Revenue ($)": rev_pred_rescaled 
#         })

# pred_df = pd.DataFrame(rows)
# print(pred_df)

# # ---------------- Save Model ----------------
# xgb_model.save_model("XGB_model\\model.json")
# print("Saved XGBoost model to XGB_model\\model.json")


# VALIDATION METRICS #
y_pred_train_norm = xgb_model.predict(X_train)
y_pred_test_norm  = xgb_model.predict(X_test)

y_train = denorm(y_train, rev_min, rev_max)
y_test  = denorm(y_test, rev_min, rev_max)
y_pred_train = denorm(y_pred_train_norm, rev_min, rev_max)
y_pred_test  = denorm(y_pred_test_norm, rev_min, rev_max)

tr_r2, tr_rmse, tr_mae, tr_mape = metrics(y_train, y_pred_train)
te_r2, te_rmse, te_mae, te_mape = metrics(y_test,  y_pred_test)

print(f"TRAIN -> R²: {tr_r2:.3f} | RMSE: ${tr_rmse:,.2f} | MAE: ${tr_mae:,.2f} | MAPE: {tr_mape:.2f}%")
print(f"TEST  -> R²: {te_r2:.3f} | RMSE: ${te_rmse:,.2f} | MAE: ${te_mae:,.2f} | MAPE: {te_mape:.2f}%")

print("Generalization gap (RMSE test/train):", round(te_rmse / tr_rmse, 2))

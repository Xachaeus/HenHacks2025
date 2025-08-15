import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
from dateutil import parser
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
from helpers import load_preprocessed_data, metrics, denorm
from MLP import MLP
import os


"""
Best model is trained on daily intervals with a test_size of 0.2 ###
Generalizes best to test data, best model, probably super overfit
"""

# df = load_raw_data("raw_dataset.json")
df = load_preprocessed_data("JSONs\\labeled_instantiated_dataset_8.json")
print(df.shape)

# ---------------- Features ----------------
X_cat = df[["school_level", "business_type", "school_type"]].values
X_num = df[["operating_time", "num_teachers", "num_students", "average_income"]].values.astype(float)
y_revenue = df["daily_revenue"].values.reshape(-1,1).astype(float)

# Normalize revenue
rev_min, rev_max = y_revenue.min(), y_revenue.max()
y_revenue_norm = (y_revenue - rev_min) / (rev_max - rev_min)

# Encode categorical features
encoder = OneHotEncoder(sparse_output=False)
X_cat_enc = encoder.fit_transform(X_cat)
X = np.hstack([X_cat_enc, X_num])

# Save encoder and min/max for inference
joblib.dump(encoder, "MLP_model\\encoder.pkl")
joblib.dump((rev_min, rev_max), "MLP_model\\rev_min_max.pkl")

# ---------------- Train/Test Split ----------------
X_train, X_test, y_rev_train, y_rev_test = train_test_split(
    X, y_revenue_norm, test_size=0.1, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_rev_train = torch.tensor(y_rev_train, dtype=torch.float32)
y_rev_test = torch.tensor(y_rev_test, dtype=torch.float32)
# y_surv_train = torch.tensor(y_surv_train, dtype=torch.float32)
# y_surv_test = torch.tensor(y_surv_test, dtype=torch.float32)

model = MLP(input_dim=X_train.shape[1])

# ---------------- Training ----------------
criterion_rev = nn.MSELoss()
criterion_surv = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-4)
loss_weight_rev = 1
num_epochs = 10000

for epoch in range(num_epochs):
    optimizer.zero_grad()
    rev_pred = model(X_train)
    loss = loss_weight_rev * criterion_rev(rev_pred, y_rev_train) 
    loss.backward()
    optimizer.step()
    scheduler.step() 

    if (epoch+1) % 50 == 0:
        with torch.no_grad():
            rev_test = model(X_test)
            test_loss = loss_weight_rev * criterion_rev(rev_test, y_rev_test) 
            
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

# ----- Generate predictions table -----
# rows = []
# with torch.no_grad():
#     for school in df["school_level"].unique():
#         for btype in df["business_type"].unique():
#             avg_time = df["operating_time"].mean()  # average operating time for predictions
#             ex_cat = encoder.transform([[school, btype]])
#             ex_num = np.array([[avg_time]])
#             ex_feat = np.hstack([ex_cat, ex_num])
#             ex_tensor = torch.tensor(ex_feat, dtype=torch.float32)

#             rev_pred = model(ex_tensor)
#             rev_rescaled = rev_pred.item() * (rev_max - rev_min) + rev_min

#             rows.append({
#                 "School": school,
#                 "Business Type": btype,
#                 "Predicted Annual Revenue ($)": rev_rescaled * 365,
#                 "Predicted Daily Revenue ($)": rev_rescaled 
#             })

# pred_df = pd.DataFrame(rows)
# print(pred_df)

# ---------------- Save Model ----------------
torch.save(model.state_dict(), "MLP_model\\model.pth")
print("Saved model to model.pth")

# VALIDATION METRICS #
y_pred_train_norm = model(X_train)
y_pred_test_norm  = model(X_test)

y_train = denorm(y_rev_train, rev_min, rev_max).detach().numpy()
y_test  = denorm(y_rev_test, rev_min, rev_max).detach().numpy()
y_pred_train = denorm(y_pred_train_norm, rev_min, rev_max).detach().numpy()
y_pred_test  = denorm(y_pred_test_norm, rev_min, rev_max).detach().numpy()

tr_r2, tr_rmse, tr_mae, tr_mape = metrics(y_train, y_pred_train)
te_r2, te_rmse, te_mae, te_mape = metrics(y_test,  y_pred_test)

print(f"TRAIN -> R²: {tr_r2:.3f} | RMSE: ${tr_rmse:,.2f} | MAE: ${tr_mae:,.2f} | MAPE: {tr_mape:.2f}%")
print(f"TEST  -> R²: {te_r2:.3f} | RMSE: ${te_rmse:,.2f} | MAE: ${te_mae:,.2f} | MAPE: {te_mape:.2f}%")

print("Generalization gap (RMSE test/train):", round(te_rmse / tr_rmse, 2))

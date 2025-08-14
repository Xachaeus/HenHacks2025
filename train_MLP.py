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
from load_json import load_data
from MLP import MLP

df = load_data("raw_dataset.json")

# ---------------- Features ----------------
X_cat = df[["school_level", "business_type"]].values
X_num = df[["avg_operating_time"]].values.astype(float)
y_revenue = df["annual_revenue"].values.reshape(-1,1).astype(float)
# y_survival = df[["survive_1mo","survive_3mo","survive_1yr"]].values.astype(float)

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
    X, y_revenue_norm, test_size=0.2, random_state=42
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
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-4)
loss_weight_rev = 0.5
num_epochs = 1000

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
rows = []
with torch.no_grad():
    for school in df["school_level"].unique():
        for btype in df["business_type"].unique():
            avg_time = df["avg_operating_time"].mean()  # average operating time for predictions
            ex_cat = encoder.transform([[school, btype]])
            ex_num = np.array([[avg_time]])
            ex_feat = np.hstack([ex_cat, ex_num])
            ex_tensor = torch.tensor(ex_feat, dtype=torch.float32)

            rev_pred = model(ex_tensor)
            rev_rescaled = rev_pred.item() * (rev_max - rev_min) + rev_min

            rows.append({
                "School": school,
                "Business Type": btype,
                "Predicted Annual Revenue ($)": rev_rescaled,
                "Predicted Daily Revenue ($)": rev_rescaled/365
            })

pred_df = pd.DataFrame(rows)
print(pred_df)

# ---------------- Save Model ----------------
torch.save(model.state_dict(), "MLP_model\\model.pth")
print("Saved model to model.pth")

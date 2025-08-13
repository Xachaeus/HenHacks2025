import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from dateutil import parser


def parse_amount(a):
    return float(a.replace("$","").replace(",","").strip())

# ----- Load JSON & preprocess -----
def load_data(filename):
    data = json.load(open(filename))
    rows = []

    for loc_id, loc_data in data.items():
        meta = loc_data["metadata"]
        transactions = loc_data.get("transactions", [])
        if not transactions:
            continue

        amounts = np.array([parse_amount(t["amount"]) for t in transactions])
        dates = [parser.parse(t["date"]) for t in transactions]

        total_revenue = amounts.sum()
        first_date, last_date = min(dates), max(dates)
        avg_operating_time = (last_date - first_date).days + 1

        months_active = len(set((d.year, d.month) for d in dates))
        survive_1mo = float(months_active >= 1)
        survive_3mo = float(months_active >= 3)
        survive_1yr = float(months_active >= 12)

        rows.append({
            "school_level": meta.get("Middle/High School", "High").lower(),
            "business_type": meta.get("Business Type", "Other").lower(),
            "avg_operating_time": avg_operating_time,
            "annual_revenue": total_revenue,
            "survive_1mo": survive_1mo,
            "survive_3mo": survive_3mo,
            "survive_1yr": survive_1yr
        })

    df = pd.DataFrame(rows)
    return df

# Load your real JSON
df = load_data("raw_dataset.json")

# ----- Features and targets -----
X_cat = df[["school_level", "business_type"]].values
X_num = df[["avg_operating_time"]].values.astype(float)
y_revenue = df["annual_revenue"].values.astype(float).reshape(-1,1)
y_survival = df[["survive_1mo","survive_3mo","survive_1yr"]].values.astype(float)

# Normalize revenue
rev_min, rev_max = y_revenue.min(), y_revenue.max()
y_revenue_norm = (y_revenue - rev_min) / (rev_max - rev_min)

# Encode categorical features
encoder = OneHotEncoder(sparse_output=False)
X_cat_enc = encoder.fit_transform(X_cat)
X = np.hstack([X_cat_enc, X_num])

# Split train/test
X_train, X_test, y_rev_train, y_rev_test, y_surv_train, y_surv_test = train_test_split(
    X, y_revenue_norm, y_survival, test_size=0.2, random_state=42
)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_rev_train = torch.tensor(y_rev_train, dtype=torch.float32)
y_rev_test = torch.tensor(y_rev_test, dtype=torch.float32)
y_surv_train = torch.tensor(y_surv_train, dtype=torch.float32)
y_surv_test = torch.tensor(y_surv_test, dtype=torch.float32)

# ----- Multi-task model -----
class MultiTaskModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.rev_head = nn.Linear(16, 1)
        self.surv_head = nn.Linear(16, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.shared(x)
        rev = self.rev_head(x)
        surv = self.sigmoid(self.surv_head(x))
        return rev, surv

model = MultiTaskModel(input_dim=X_train.shape[1])

# ----- Training -----
criterion_rev = nn.MSELoss()
criterion_surv = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_weight_rev = 0.1

for epoch in range(100):
    optimizer.zero_grad()
    rev_pred, surv_pred = model(X_train)
    loss = loss_weight_rev * criterion_rev(rev_pred, y_rev_train) + criterion_surv(surv_pred, y_surv_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            rev_test, surv_test = model(X_test)
            test_loss = loss_weight_rev * criterion_rev(rev_test, y_rev_test) + criterion_surv(surv_test, y_surv_test)
        print(f"Epoch {epoch+1}/100, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

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

            rev_pred, surv_pred = model(ex_tensor)
            rev_rescaled = rev_pred.item() * (rev_max - rev_min) + rev_min

            rows.append({
                "School": school,
                "Business Type": btype,
                "Predicted Annual Revenue ($)": rev_rescaled,
                "Survival ≥1mo (%)": surv_pred[0,0].item()*100,
                "Survival ≥3mo (%)": surv_pred[0,1].item()*100,
                "Survival ≥1yr (%)": surv_pred[0,2].item()*100
            })

pred_df = pd.DataFrame(rows)
print(pred_df)

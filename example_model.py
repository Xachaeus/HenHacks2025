import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

np.random.seed(42)
n_samples = 500
school_levels_unique = ["middle", "high"]
school_businesses_unique = ["apparel", "food", "services"]

# ----- Dummy Data -----
school_levels = np.random.choice(school_levels_unique, n_samples)
business_types = np.random.choice(school_businesses_unique, n_samples)
avg_operating_time = np.random.uniform(0, 500, n_samples)  # days

# Revenue depends on school type and business type
revenue_base = np.array([3, 2, 1])  # base multiplier
revenue_school = np.array([1.0, 1.0])  # middle/high
type_idx = np.array([{"apparel":0,"food":1,"services":2}[t] for t in business_types])
school_idx = np.array([{"middle":0,"high":1}[s] for s in school_levels])
annual_revenue = avg_operating_time * revenue_base[type_idx] * revenue_school[school_idx]
annual_revenue += np.random.normal(100, 5000, n_samples)  # realistic noise

# ----- Normalize revenue to 0–1 -----
rev_min, rev_max = annual_revenue.min(), annual_revenue.max()
y_revenue_norm = ((annual_revenue - rev_min) / (rev_max - rev_min)).astype(np.float32).reshape(-1,1)

# ----- Stochastic survival labels -----
prob_1m = np.clip(0.2 + 0.6 * (avg_operating_time / 30), 0, 1)
prob_3m = np.clip(0.1 + 0.5 * (avg_operating_time / 120), 0, 1)
prob_1y = np.clip(0.05 + 0.3 * (avg_operating_time / 365), 0, 1)

survive_1m = np.random.binomial(1, prob_1m)
survive_3m = np.random.binomial(1, prob_3m)
survive_1y = np.random.binomial(1, prob_1y)
y_survival = np.column_stack([survive_1m, survive_3m, survive_1y]).astype(np.float32)

# ----- Preprocessing -----
encoder = OneHotEncoder(sparse_output=False)
X_cat = encoder.fit_transform(np.column_stack([school_levels, business_types]))
X_num = avg_operating_time.reshape(-1,1)
X = np.hstack([X_cat, X_num])

X_train, X_test, y_rev_train, y_rev_test, y_surv_train, y_surv_test = train_test_split(
    X, y_revenue_norm, y_survival, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_rev_train = torch.tensor(y_rev_train, dtype=torch.float32)
y_rev_test = torch.tensor(y_rev_test, dtype=torch.float32)
y_surv_train = torch.tensor(y_surv_train, dtype=torch.float32)
y_surv_test = torch.tensor(y_surv_test, dtype=torch.float32)

# ----- Multi-task Model -----
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
loss_weight_rev = 0.1  # reduce revenue contribution

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

# ----- Example Prediction -----
with torch.no_grad():
    example = np.array([["high","food",120]])
    example_cat = encoder.transform(example[:,:2])
    example_num = example[:,2:].astype(float)
    example_feat = np.hstack([example_cat, example_num])
    example_tensor = torch.tensor(example_feat, dtype=torch.float32)
    
    rev_pred, surv_pred = model(example_tensor)
    rev_rescaled = rev_pred.item() * (rev_max - rev_min) + rev_min
    print(f"\nPredicted Annual Revenue: ${rev_rescaled:.2f}")
    print(f"Predicted Survival ≥1mo: {surv_pred[0,0].item()*100:.1f}%")
    print(f"Predicted Survival ≥3mo: {surv_pred[0,1].item()*100:.1f}%")
    print(f"Predicted Survival ≥1yr: {surv_pred[0,2].item()*100:.1f}%")

rows = []
with torch.no_grad():
    for school in school_levels_unique:
        for btype in school_businesses_unique:
            ex_cat = encoder.transform([[school, btype]])
            ex_num = np.array([[120]]).astype(float)  # fixed operating time
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

df = pd.DataFrame(rows)
print(df)

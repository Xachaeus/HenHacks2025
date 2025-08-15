from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
from load_json import load_preprocessed_data
import pandas as pd

"""
Best model is trained on weekly intervals with a test_size of 0.1 ###
Generalizes to ~12% of test data, worst model
"""

# ---------------- Load dataset ----------------
df = load_preprocessed_data("JSONs\\preprocessed_dataset_instances_30.json")

# Include categorical + numeric feature
X = df[["school_level", "business_type", "operating_time"]]
y = df["daily_revenue"]

# ---------------- Preprocessor ----------------
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), ["school_level", "business_type"]),
        ('num', 'passthrough', ["operating_time"])  # numeric feature passthrough
    ]
)

# ---------------- Pipeline ----------------
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# ---------------- Train/Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=40
)

model.fit(X_train, y_train)

print("Train R²:", model.score(X_train, y_train))
print("Test R²:", model.score(X_test, y_test))

# ---------------- Save model ----------------
joblib.dump(model, "LR_model\\school_revenue_model.pkl")

# ---------------- Generate predictions table ----------------
school_levels = df["school_level"].unique()
business_types = df["business_type"].unique()
operating_times = [7, 30, 365]  # example time values

# Create all combinations
combinations_df = pd.DataFrame(
    [(sl, bt, ot) for sl in school_levels for bt in business_types for ot in operating_times],
    columns=["school_level", "business_type", "operating_time"]
)

# Predict
predictions = model.predict(combinations_df)

# Results table
result_df = combinations_df.copy()
result_df["predicted_daily_revenue"] = predictions
result_df["predicted_total_revenue"] = result_df["predicted_daily_revenue"] * result_df["operating_time"]

print(result_df)

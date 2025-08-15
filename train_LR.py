from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
from helpers import load_preprocessed_data, denorm, metrics
import pandas as pd

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType, FloatTensorType

"""
Best model is trained on weekly intervals with a test_size of 0.1 ###
Generalizes to ~12% of test data, worst model
"""

# ---------------- Load dataset ----------------
df = load_preprocessed_data("JSONs\\labeled_instantiated_dataset_8.json")

# Include categorical + numeric feature
X = df[["school_level", "business_type", "school_type", 
        "operating_time", "num_teachers", "num_students", "average_income"]]
y = df["daily_revenue"]

# ---------------- Preprocessor ----------------
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), ["school_level", "business_type", "school_type"]),
        ('num', 'passthrough', ["operating_time", "num_teachers", "num_students", "average_income"])
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
# predictions = model.predict(combinations_df)

# # Results table
# result_df = combinations_df.copy()
# result_df["predicted_daily_revenue"] = predictions
# result_df["predicted_total_revenue"] = result_df["predicted_daily_revenue"] * result_df["operating_time"]

# print(result_df)

# VALIDATION METRICS #
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

# y_train = denorm(y_train, rev_min, rev_max)
# y_test  = denorm(y_test, rev_min, rev_max)
# y_pred_train = denorm(y_pred_train_norm, rev_min, rev_max)
# y_pred_test  = denorm(y_pred_test_norm, rev_min, rev_max)

tr_r2, tr_rmse, tr_mae, tr_mape = metrics(y_train, y_pred_train)
te_r2, te_rmse, te_mae, te_mape = metrics(y_test,  y_pred_test)

print(f"TRAIN -> R²: {tr_r2:.3f} | RMSE: ${tr_rmse:,.2f} | MAE: ${tr_mae:,.2f} | MAPE: {tr_mape:.2f}%")
print(f"TEST  -> R²: {te_r2:.3f} | RMSE: ${te_rmse:,.2f} | MAE: ${te_mae:,.2f} | MAPE: {te_mape:.2f}%")

print("Generalization gap (RMSE test/train):", round(te_rmse / tr_rmse, 2))

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
from load_json import load_preprocessed_data
import pandas as pd

# Load dataset
df = load_preprocessed_data("preprocessed_dataset_instances_7.json")

# Only categorical features
X = df[["school_level", "business_type"]]
y = df["daily_revenue"]

# Build preprocessing + regression pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), ["school_level", "business_type"])
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.7, random_state=40
)

model.fit(X_train, y_train)

print("Train R²:", model.score(X_train, y_train))
print("Test R²:", model.score(X_test, y_test))

# Save model for Streamlit
joblib.dump(model, "LR_model\\school_revenue_model.pkl")

# Get unique values from the dataset
school_levels = df["school_level"].unique()
business_types = df["business_type"].unique()

# Create all combinations
combinations_df = pd.DataFrame(
    [(sl, bt) for sl in school_levels for bt in business_types],
    columns=["school_level", "business_type"]
)

# Predict
predictions = model.predict(combinations_df)

# Results
result_df = combinations_df.copy()
result_df["expected_revenue"] = predictions
print(result_df)

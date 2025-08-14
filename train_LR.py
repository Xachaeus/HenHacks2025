from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
from load_json import load_data
import pandas as pd

# Load dataset
df = load_data("raw_dataset.json")

# Only categorical features
# X = df[["school_level", "business_type"]]
X = df[["school_level"]]
y = df["annual_revenue"]

# Build preprocessing + regression pipeline
preprocessor = ColumnTransformer(
    transformers=[
        # ('cat', OneHotEncoder(drop='first'), ["school_level", "business_type"])
        ('cat', OneHotEncoder(drop='first'), ["school_level"])
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=40
)

model.fit(X_train, y_train)

print("Train R²:", model.score(X_train, y_train))
print("Test R²:", model.score(X_test, y_test))

# Save model for Streamlit
joblib.dump(model, "school_revenue_model.pkl")

# Get unique categories from the original dataframe
school_levels = df["school_level"].unique()

# Create all combinations (just one column in this case)
combinations_df = pd.DataFrame({"school_level": school_levels})

# Predict expected revenue
predictions = model.predict(combinations_df)

# Show results
result_df = combinations_df.copy()
result_df["expected_revenue"] = predictions
print(result_df)

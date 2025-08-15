import json
import pandas as pd
import numpy as np

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
import numpy as np
from math import ceil

# --- Helper to unnormalize back to dollars ---
def denorm(y, rev_min, rev_max):
    return y.ravel() * (rev_max - rev_min) + rev_min

# --- Metrics ---
def metrics(y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    # Safe MAPE (ignores zeros)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    return r2, rmse, mae, mape

# ---------------- Load JSON ----------------
def parse_amount(a):
    return float(a.replace("$","").replace(",","").strip())

def load_preprocessed_data(filename):
    # Load JSON
    data = json.load(open(filename))
    granularity = int(filename[-6])

    rows = []
    for instance in data:
        average_income = instance['average income']
        if average_income == '':
            average_income = 50000
        num_students = int(instance['Number of Students'])
        num_teachers = int(instance['Number of Teachers'])
        school_type = instance['Public/ Private']
        # Convert amounts and dates
        total_revenue = int(instance['scaled total'])
        operating_time = int(ceil(instance['search window']))
        
        annual_revenue = total_revenue / (granularity / 365)
        daily_revenue = total_revenue / granularity
        
        # Survival flags
        # months_active = len(set((d.year, d.month) for d in dates))
        # survive_1mo = bool(months_active >= 1)
        # survive_3mo = bool(months_active >= 3)
        # survive_1yr = bool(months_active >= 12)
        
        if total_revenue < 1000000:
            rows.append({
                "school_level": instance["Middle/High School"].lower(),
                "business_type": instance["Business Type"].lower(),
                "operating_time": operating_time,
                "annual_revenue": annual_revenue,
                "daily_revenue": daily_revenue,
                "average_income": average_income,
                "num_students": num_students,
                "num_teachers": num_teachers,
                "school_type": school_type
                # "survive_1mo": survive_1mo,
                # "survive_3mo": survive_3mo,
                # "survive_1yr": survive_1yr
            })

    df = pd.DataFrame(rows)
    df = df[df["school_level"].isin(["middle","high"])]
    df.reset_index(drop=True, inplace=True)
    
    return df

        
# load_preprocessed_data("preprocessed_dataset_instances_7.json")
# load_raw_data("raw_dataset.json")


### Look for negatives in culinary ###
# df = load_preprocessed_data("JSONs\\labeled_instantiated_dataset_30.json")
# print(df["business_type"].unique)
# df = df[df["business_type"].isin(["culinary shop"])]
# df.reset_index(drop=True, inplace=True)
# X = df[["school_level", "business_type", "operating_time", "daily_revenue"]]

# print(X)

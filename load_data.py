import json
import pandas as pd
import numpy as np
from datetime import datetime

def load_data(filename):
    # Load JSON
    data = json.load(open(filename))

    rows = []
    for loc_id, loc_data in data.items():
        meta = loc_data["metadata"]
        transactions = loc_data["transactions"]
        if not transactions:
            continue
        
        # Convert amounts and dates
        amounts = np.array([float(t["amount"]) for t in transactions])
        dates = [datetime.strptime(t["date"], "%m/%d/%Y") for t in transactions]
        
        total_revenue = amounts.sum()
        first_date, last_date = min(dates), max(dates)
        avg_operating_time = (last_date - first_date).days + 1
        
        # Survival flags
        months_active = len(set((d.year, d.month) for d in dates))
        survive_1mo = bool(months_active >= 1)
        survive_3mo = bool(months_active >= 3)
        survive_1yr = bool(months_active >= 12)
        
        rows.append({
            "school_level": meta.get("Middle/High School", "High"),
            "business_type": meta.get("Business Type", "Other"),
            "avg_operating_time": avg_operating_time,
            "annual_revenue": total_revenue,
            "survive_1mo": survive_1mo,
            "survive_3mo": survive_3mo,
            "survive_1yr": survive_1yr
        })

    # df = pd.DataFrame(rows)
    # print(df.head())
    
    return data

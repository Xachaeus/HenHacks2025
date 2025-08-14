import json
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser


# ---------------- Load JSON ----------------
def parse_amount(a):
    return float(a.replace("$","").replace(",","").strip())


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
        amounts = np.array([parse_amount(t["amount"]) for t in transactions if parse_amount(t["amount"]) < 1000 and parse_amount(t["amount"]) > -1000])
        dates = [parser.parse(t["date"]) for t in transactions]        
        total_revenue = amounts.sum()
        first_date, last_date = min(dates), max(dates)
        avg_operating_time = (last_date - first_date).days + 1
        
        annual_revenue = total_revenue / (avg_operating_time / 365)
        daily_revenue = total_revenue / avg_operating_time
        
        # Survival flags
        # months_active = len(set((d.year, d.month) for d in dates))
        # survive_1mo = bool(months_active >= 1)
        # survive_3mo = bool(months_active >= 3)
        # survive_1yr = bool(months_active >= 12)
        
        if total_revenue < 1000000:
            rows.append({
                "school_level": meta.get("Middle/High School", "High").lower(),
                "business_type": meta.get("Business Type", "Other").lower(),
                "avg_operating_time": avg_operating_time,
                "annual_revenue": annual_revenue,
                "daily_revenue": daily_revenue,
                # "survive_1mo": survive_1mo,
                # "survive_3mo": survive_3mo,
                # "survive_1yr": survive_1yr
            })

    df = pd.DataFrame(rows)
    df = df[df["school_level"].isin(["middle","high"])]
    df.reset_index(drop=True, inplace=True)
    
    return df

import os, csv, pickle
from datetime import datetime, timedelta
import json

DATASET_SRC_DIR = "./dataset/Data/"
DATASET_SAVETO = "./raw_dataset.json"
DO_SCRAPING = True

# Below are some identified uncategorized location ids and their manually-determined valid alternatives
# All misnamed transactions come from the files:
# ./dataset/Data/Square Transaction Data\transactions-2024-07-01-2025-07-01 (13).csv
# ./dataset/Data/Square Transaction Data\transactions-2023-07-01-2024-07-01 (14).csv
ALTERNATIVE_LOCATION_NAMES = {
    "Cavs Mart (old location)": "Cavs Mart",
    "Odyssey Charter School":                   "Odyssey Charter School (BPA/StudentCouncil)",
    "AHS Café":                                 "dzPgMMEnQga6UCpB8j0n",
    "Odessa High School School Store":          "", # No likely replacement found, discarding
    "Sussex Central High School KnightWares":   "SCHS CTE Store",
    "Sips":                                     "", # No likely replacement found, discarding
    "Sweet Escape Candy Company":               "", # No likely replacement found, discarding
    "Exotic Eats and Treats":                   "", # No likely replacement found, discarding
    "Brewtopia":                                "", # No likely replacement found, discarding
    "Reminisce":                                "", # No likely replacement found, discarding
    "Cab Calloway Communication Arts":          "HrRh9J8WDIKGP3GA78bO",
    "Alexis i duPont High School":              "AIHS School Store - Tigers Den"
}

def print_dataset(dataset):
    print("")
    for company in dataset.keys():
        print(f"{dataset[company]["metadata"]["School"]}: {len(dataset[company]["transactions"])} transactions, ${dataset[company]["metadata"]["total profit"]:.2f} made in {dataset[company]["metadata"]["duration"]} days starting in {dataset[company]["metadata"]["launch month"]}")
    print("")


if DO_SCRAPING:
    extracted_files = []

    # Begin by fetching location data from provided XLSX (converted to csv for ease of use)
    with open("./dataset/School_Location_ID_Mapping.csv", 'r', encoding="UTF-8") as business_csv:
        business_file = csv.reader(business_csv)
        column_headers = next(business_file)
        business_data = {}
        for row in business_file:
            curr_data = {}
            for header, value in zip(column_headers, row):
                curr_data.update({header: value})
            business_data.update({curr_data["Location (PK)"]: curr_data})

    print("Searching for CSVs...")

    # Find all csv files in the Data subdirectory
    for (root, dirs, files) in os.walk(DATASET_SRC_DIR):
        for filepath in files:

            if filepath.endswith(".csv"):
                
                with open(os.path.join(root, filepath), 'r', encoding="UTF-8") as csv_file: 


                    csv_reader = csv.reader(csv_file)
                    fields = next(csv_reader)

                    file_data = []

                    # Create a dictionary for every transaction found
                    for row in csv_reader:
                        data = {title:None for title in fields}
                        for column, field in zip(row, fields):
                            data[field] = column
                        data.update({"src": os.path.join(root, filepath)})
                        file_data.append(data)

                    extracted_files.append(file_data)

    print("Done!\nReformatting transaction data...")

    # Reformat the data into one universal format that can be easily accessed later
    extracted_data = []
    for file_data in extracted_files:
        for data in file_data:
            try:
                # Format to use for Square data
                extracted_data.append({
                    "location": data["Location"],
                    "amount": data["Net Total"],
                    "date": data["Date"],
                })

                # Debugging print statements for erroneous location ids
                #if data["Location"] == "Exotic Eats and Treats": print(data["src"])
                #if data["Location"] in ALTERNATIVE_LOCATION_NAMES.keys(): 
                    #if ALTERNATIVE_LOCATION_NAMES[data["Location"]] == "": print(f"{data['src']}")

            except:
                # Format to use for Stripe data
                try:
                    extracted_data.append({
                        "location": data["locationId (metadata)"],
                        "amount": data["Amount"],
                        "date": data["Created date (UTC)"].split(' ')[0]
                    })
                except:
                    extracted_data.append({
                        "location": data["altId (metadata)"],
                        "amount": data["Amount"],
                        "date": data["Created date (UTC)"].split(' ')[0]
                    })

    print("Done!")

    # Conglomerate the transaction data into something useful
    print("Organizing data...")
    dataset = {}
    for data in extracted_data:
        
        # Check for mislabeled locations
        if data["location"] in ALTERNATIVE_LOCATION_NAMES.keys(): data["location"] = ALTERNATIVE_LOCATION_NAMES[data["location"]]

        # Skip empty location values
        if data["location"] == '': continue

        # If an alternative location was provided, detect this and substitute the primary location
        for profile in business_data.values():
            if data["location"] == profile["Location2"] or data["location"] == profile["Location3"]:
                data["location"] = profile["Location (PK)"]
        # Add the transaction to the corresponding location's collection
        if data["location"] not in dataset.keys(): dataset.update({data["location"]: {"metadata": business_data[data["location"]], "transactions": [data]}})
        else: dataset[data["location"]]["transactions"].append(data)

    print("Done!")

    with open(DATASET_SAVETO, 'w') as f: json.dump(dataset, f)


print("Preprocessing data and extracting cumulative features...")
with open(DATASET_SAVETO, 'r') as f: dataset = json.load(f)


business_data = {}
for location, data in dataset.items():
    dates = [(datetime.strptime(transaction_date, "%Y-%m-%d") if '/' not in transaction_date else datetime.strptime(transaction_date, "%m/%d/%Y")) for transaction_date in [x["date"] for x in data["transactions"]]]
    earliest = min(dates)
    latest = max(dates)
    dataset[location]["metadata"].update({"duration": (latest-earliest).days})

    valid_dates = [date for date in dates if date < (earliest + timedelta(days=365))]
    dataset[location]["metadata"].update({"valid duration": (latest-earliest).days})

    valid_transactions = dates = [transaction for transaction in data["transactions"] if (datetime.strptime(transaction["date"], "%Y-%m-%d") if '/' not in transaction["date"] else datetime.strptime(transaction["date"], "%m/%d/%Y"))]
    valid_total = sum([float(x["amount"].replace('$','').replace(',','')) for x in valid_transactions])
    dataset[location]["metadata"].update({"valid total": valid_total})
    dataset[location]["metadata"].update({"num valid transactions": len(valid_transactions)})

    total = sum([float(x["amount"].replace('$','').replace(',','')) for x in data["transactions"]])
    dataset[location]["metadata"].update({"total profit": total})

    launch_month = int(earliest.strftime("%m"))
    dataset[location]["metadata"].update({"launch month": launch_month})

    if dataset[location]["metadata"]["Business Type"] not in business_data.keys(): business_data.update({dataset[location]["metadata"]["Business Type"]: [(latest-earliest).days]})
    else: business_data[dataset[location]["metadata"]["Business Type"]].append((latest-earliest).days)

business_average_times = {}
for category, duration in business_data.items():
    business_average_times.update({category: float(sum(duration))/float(len(duration))})

for school, data in dataset:
    dataset[school]["metadata"].update({"average duration": business_average_times[data["metadata"]["Business Type"]]})

with open('preprocessed_dataset.json', 'w') as f: json.dump(dataset, f)
print("Done!")

print(business_average_times)
print_dataset(dataset)
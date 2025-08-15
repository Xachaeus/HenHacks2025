import os, csv, pickle
from datetime import datetime, timedelta
import json
from tqdm import tqdm

DATASET_SRC_DIR = "./dataset/Data/"
DATASET_SAVETO = "./raw_dataset.json"
DO_SCRAPING = True
INSTANCE_GRANULARITY = 1

# Below are some identified uncategorized location ids and their manually-determined valid alternatives
# All misnamed transactions come from the files:
# ./dataset/Data/Square Transaction Data\transactions-2024-07-01-2025-07-01 (13).csv
# ./dataset/Data/Square Transaction Data\transactions-2023-07-01-2024-07-01 (14).csv
ALTERNATIVE_LOCATION_NAMES = {
    "Cavs Mart (old location)": "Cavs Mart",
    "Odyssey Charter School":                   "Odyssey Charter School (BPA/StudentCouncil)",
    "AHS Café":                                 "dzPgMMEnQga6UCpB8j0n",
    "Odessa High School School Store":          "Odessa High School School Store", # No likely replacement found, discarding
    "Sussex Central High School KnightWares":   "SCHS CTE Store",
    "Sips":                                     "Odessa High School School Store", # No likely replacement found, discarding
    "Sweet Escape Candy Company":               "Odessa High School School Store", # No likely replacement found, discarding
    "Exotic Eats and Treats":                   "", # No likely replacement found, discarding
    "Brewtopia":                                "Brewtopia", # No likely replacement found, discarding
    "Reminisce":                                "", # No likely replacement found, discarding
    "Smashing Blooms by SITE":                  "",
    "vday":                                     "",
    "Cab Calloway Communication Arts":          "HrRh9J8WDIKGP3GA78bO",
    "Alexis i duPont High School":              "AIHS School Store - Tigers Den"
}

def print_dataset(dataset):
    print("")
    for company in dataset.keys():
        print(
            f"{company}: {len(dataset[company]['instances'])} instances, "
            f"{len(dataset[company]['transactions'])} transactions, "
            f"${dataset[company]['metadata']['total profit']:.2f} made in "
            f"{dataset[company]['metadata']['duration']} days starting in "
            f"{dataset[company]['metadata']['launch month']}"
        )
    print("")


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

print("Searching for CSVs... ", end="")

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

print("Done!\nReformatting transaction data... ", end="")

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
                "src": data["src"]
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
                    "date": data["Created date (UTC)"].split(' ')[0],
                    "src": data["src"]
                })
            except:
                extracted_data.append({
                    "location": data["altId (metadata)"],
                    "amount": data["Amount"],
                    "date": data["Created date (UTC)"].split(' ')[0],
                    "src": data["src"]
                })

print("Done!")

# Conglomerate the transaction data into something useful
print("Organizing data... ", end="")

dataset = {business_data[location]['Location (PK)']: {'metadata': business_data[location], 'transactions': []} for location in business_data.keys()}
for data in extracted_data:

    if abs(float(data["amount"].replace('$','').replace(',',''))) > 10000: continue
    
    # Check for mislabeled locations
    if data["location"] in ALTERNATIVE_LOCATION_NAMES.keys(): data["location"] = ALTERNATIVE_LOCATION_NAMES[data["location"]]

    # Skip empty location values
    if data["location"] == '': continue

    # If an alternative location was provided, detect this and substitute the primary location
    for profile in business_data.values():
        if data["location"] == profile["Location2"] or data["location"] == profile["Location3"]:
            data["location"] = profile["Location (PK)"]
    # Add the transaction to the corresponding location's collection
    if data["location"] not in dataset.keys(): # Remove schools with no transaction data
        continue
    dataset[data["location"]]["transactions"].append(data)

print("Done!")
print("Filtering out locations with no transactions... ", end="")

good_data = {}
for location in dataset.keys():
    if len(dataset[location]['transactions']) == 0: 
        #print(f"{dataset[location]['metadata']['School']}: {dataset[location]['metadata']['Location2']}")
        pass
    else:
        good_data.update({location: dataset[location]})

dataset = good_data

print("Done!")
print("Preprocessing data and extracting cumulative features...")


business_data = {}
for location, data in tqdm(dataset.items()):

    dates = []
    potential_transactions = []
    for transaction in data["transactions"]:
        transaction_date = transaction["date"]
        if '/' not in transaction_date:
            dates.append(datetime.strptime(transaction_date, "%Y-%m-%d"))
            potential_transactions.append((transaction, datetime.strptime(transaction_date, "%Y-%m-%d")))
        elif len(transaction_date.split('/')[-1]) > 2:
            dates.append(datetime.strptime(transaction_date, "%m/%d/%Y"))
            potential_transactions.append((transaction, datetime.strptime(transaction_date, "%m/%d/%Y")))
        else:
            dates.append(datetime.strptime(transaction_date, "%m/%d/%y"))
            potential_transactions.append((transaction, datetime.strptime(transaction_date, "%m/%d/%y")))

    earliest = min(dates)
    latest = max(dates)
    dataset[location]["metadata"].update({"duration": (latest-earliest).days+1})

    total = sum([float(x["amount"].replace('$','').replace(',','')) for x in data["transactions"]])
    dataset[location]["metadata"].update({"total profit": total})

    launch_month = int(earliest.strftime("%m"))
    dataset[location]["metadata"].update({"launch month": launch_month})
    
    instances = []

    prev_time_window = 0
    time_window = INSTANCE_GRANULARITY/2.0

    could_continue = True
    while could_continue:

        if (earliest + timedelta(time_window)) >= latest: could_continue = False

        # All school-associated metadata should be grandfathered in here
        current_instance = {"Business Type": dataset[location]["metadata"]["Business Type"],
                            "Middle/High School": dataset[location]["metadata"]["Middle/High School"],
                            "average income": dataset[location]["metadata"]["Average Yearly Income per Household"],
                            "Public/ Private": dataset[location]["metadata"]["Public/ Private"],
                            "Number of Students": dataset[location]["metadata"]["Number of Students"],
                            "Number of Teachers": dataset[location]["metadata"]["Number of Teachers"]}

        valid_dates = [date for date in dates if date >= (earliest + timedelta(days=prev_time_window)) and date <= (earliest + timedelta(days=time_window))]
        #if len(valid_dates) != 0:
        if (len(valid_dates) != 0):
            current_instance.update({"valid duration": (max(valid_dates)-min(valid_dates)).days+1})
        else:
            current_instance.update({"valid duration": 0})
        current_instance.update({"could continue": could_continue})

        valid_transactions = [transaction for transaction, transaction_date in potential_transactions if transaction_date >= (earliest + timedelta(days=prev_time_window)) and transaction_date <= (earliest + timedelta(days=time_window))]
        #valid_total = sum([float(x["amount"].replace('$','').replace(',','')) for x in valid_transactions])
        #current_instance.update({"valid total": valid_total})
        current_instance.update({"num valid transactions": len(valid_transactions)})

        #scaled_transactions = [ (float(transaction["amount"].replace('$','').replace(',',''))) * (1/abs((earliest + timedelta(days=prev_time_window))-transaction_date)) for transaction, transaction_date in potential_transactions]
        scaled_total = 0
        for transaction, transaction_date in potential_transactions:
            amount = float(transaction["amount"].replace('$','').replace(',',''))
            dist = (abs(((earliest + timedelta(days=prev_time_window))-transaction_date).days)) / (INSTANCE_GRANULARITY/2)
            if dist > 1: scale = 1.0/float(dist)
            else: scale = 1.0
            scaled_total += amount * scale
        current_instance.update({"scaled total": scaled_total})


        current_instance.update({"launch month": launch_month})
        current_instance.update({"search window": time_window})

        instances.append(current_instance)

        prev_time_window = time_window
        time_window += INSTANCE_GRANULARITY
        

    dataset[location].update({"instances": instances})

    if dataset[location]["metadata"]["Business Type"] not in business_data.keys(): business_data.update({dataset[location]["metadata"]["Business Type"]: [(latest-earliest).days]})
    else: business_data[dataset[location]["metadata"]["Business Type"]].append((latest-earliest).days)

print("Done!")
print("Formatting into usable datasets... ", end="")

business_average_times = {}
for category, duration in business_data.items():
    business_average_times.update({category: float(sum(duration))/float(len(duration))})

business_types = list(business_average_times.keys())
for school, data in dataset.items():
    for idx, instance in enumerate(dataset[school]['instances']):
        dataset[school]["instances"][idx].update({"type id": business_types.index(dataset[school]["instances"][idx]["Business Type"])/len(business_average_times)})

instantiated_dataset = [data["instances"] for school, data in dataset.items()]

labeled_instantiated_dataset_components = [(data["metadata"]["School"], data["instances"]) for school, data in dataset.items()]
labeled_instantiated_dataset = []
for school, instances in labeled_instantiated_dataset_components:
    for instance in instances:
        instance.update({"School": school})
    labeled_instantiated_dataset += instances

human_readable_dataset = [data["metadata"] for school, data in dataset.items()]

with open('human_readable_dataset.json', 'w') as f: json.dump(human_readable_dataset, f, indent=4)
with open('labeled_instantiated_dataset.json', 'w') as f: json.dump(labeled_instantiated_dataset, f, indent=4)
with open('preprocessed_dataset.json', 'w') as f: json.dump([list(business_average_times.keys()), dataset], f)
with open('preprocessed_dataset_instances.json', 'w') as f: json.dump(instantiated_dataset, f)
print("Done!")

# print_dataset(dataset)

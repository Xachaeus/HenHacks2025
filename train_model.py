import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import json


def preprocess_instance(instance, categories):

    metadata = instance["metadata"]
    transactions = instance["transactions"]

    # Outputs
    transaction_total = metadata["valid total"]
    succeeded_after_year = int(metadata["duration"]) > 365
    operating_time = int(metadata["duration"])
    num_of_transactions = metadata["num valid transactions"]

    transaction_total_tensor = torch.tensor([int(bit) for bit in format(transaction_total, '032b')])
    operating_time_tensor = torch.tensor([int(bit) for bit in format(operating_time, '032b')])
    num_of_transactions_tensor = torch.tensor([int(bit) for bit in format(num_of_transactions, '032b')])
    succeeded_after_year_tensor = torch.tensor([1.0 if succeeded_after_year else 0.0])

    output_tensor = torch.cat((transaction_total_tensor, operating_time_tensor, num_of_transactions_tensor, succeeded_after_year_tensor), dim=0)
    
    # Inputs
    average_duration = metadata["average duration"]
    category = metadata["Business Type"]
    isHighschool = metadata["Middle/High School"] == "High"
    launch_month = metadata["launch month"]

    average_duration_tensor = torch.tensor([int(bit) for bit in format(average_duration, '032b')])
    category_tensor = torch.tensor([(1.0 if category == c else 0.0) for c in categories])
    launch_month_tensor = torch.tensor([(1.0 if launch_month == num else 0.0) for num in range(1,13)])
    isHighschool_tensor = torch.tensor([1.0 if isHighschool else 0.0])

    input_tensor = torch.cat((average_duration_tensor, category_tensor, launch_month_tensor, isHighschool_tensor), dim=0)

    return input_tensor, output_tensor


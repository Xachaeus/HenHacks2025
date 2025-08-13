import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import json


def preprocess_instance(metadata):

    # Outputs
    transaction_total = metadata["valid total"]
    succeeded_after_year = int(metadata["duration"]) > 365

    # Inputs
    average_duration = metadata["average duration"]
    category = metadata["Business Type"]
    isHighschool = metadata["Middle/High School"] == "High"
    launch_month = metadata["launch month"]
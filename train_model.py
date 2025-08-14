import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from tqdm import tqdm
import json


def preprocess_instance(instance, categories):

    metadata = instance["metadata"]

    # Outputs
    transaction_total = int(metadata["valid total"])
    succeeded_after_year = int(metadata["duration"]) > 365
    operating_time = int(metadata["duration"])
    num_of_transactions = int(metadata["num valid transactions"])

    transaction_total_tensor = torch.tensor([int(bit) for bit in format(transaction_total, '032b')])
    operating_time_tensor = torch.tensor([int(bit) for bit in format(operating_time, '032b')])
    num_of_transactions_tensor = torch.tensor([int(bit) for bit in format(num_of_transactions, '032b')])
    succeeded_after_year_tensor = torch.tensor([1.0 if succeeded_after_year else 0.0])

    output_tensor = torch.cat((transaction_total_tensor, operating_time_tensor, num_of_transactions_tensor, succeeded_after_year_tensor), dim=0)
    
    # Inputs
    average_duration = int(metadata["average duration"])
    category = metadata["Business Type"]
    isHighschool = metadata["Middle/High School"] == "High"
    launch_month = int(metadata["launch month"])

    average_duration_tensor = torch.tensor([int(bit) for bit in format(average_duration, '032b')])
    category_tensor = torch.tensor([(1.0 if category == c else 0.0) for c in categories])
    launch_month_tensor = torch.tensor([(1.0 if launch_month == num else 0.0) for num in range(1,13)])
    isHighschool_tensor = torch.tensor([1.0 if isHighschool else 0.0])

    input_tensor = torch.cat((average_duration_tensor, category_tensor, launch_month_tensor, isHighschool_tensor), dim=0)

    return input_tensor, output_tensor


def extract_output(output_tensor):

    output_tensor = torch.tensor([(1.0 if idx.item() > 0.5 else 0.0) for idx in output_tensor])
    output_arguments = [[int(x) for x in tensor.tolist()] for tensor in torch.tensor_split(output_tensor, [32,64,96], dim=0)]

    transaction_total = int("".join(map(str, output_arguments[0])), 2)
    operating_time = int("".join(map(str, output_arguments[1])), 2)
    num_transactions = int("".join(map(str, output_arguments[2])), 2)
    succeeded_after_year = output_arguments[3][0] == 1

    return (transaction_total, operating_time, num_transactions, succeeded_after_year)



class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_ff):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, d_ff),
            nn.Linear(d_ff, d_ff),
            nn.Linear(d_ff, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.shared(x)
    



if __name__ == "__main__":

    AVERAGES = 5
    EPOCHS = 300
    LEARNING_RATE = 1e-3
    device = 'cuda'

    with open("preprocessed_dataset.json", 'r') as f: categories, dataset = json.load(f)

    dataset = dataset.values()

    train_dataset = [x for idx, x in enumerate(dataset) if idx%2]
    test_dataset = [x for idx, x in enumerate(dataset) if not idx%2]

    train_tensor_dataset = [preprocess_instance(instance, categories) for instance in train_dataset]
    test_tensor_dataset = [preprocess_instance(instance, categories) for instance in test_dataset]


    scores = [0,0,0,0]
    num_tested = 0
    
    for iteration in range(AVERAGES):
        model = MultiTaskModel(train_tensor_dataset[0][0].size(0), train_tensor_dataset[0][1].size(0), 128).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        losses = []
        print("Beginning training...")
        # Train
        model.train()
        for epoch in tqdm(range(EPOCHS)):
            for input_tensor, expected_tensor in train_tensor_dataset:
                optimizer.zero_grad()
                input_tensor = input_tensor.to(device)
                expected_tensor = expected_tensor.to(device)

                output = model(input_tensor)
                loss = criterion(output, expected_tensor)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        print("Done!\nBeginning testing...")
        # Test
        model.eval()
        with torch.no_grad():
            for input_tensor, expected_tensor in test_tensor_dataset:
                input_tensor = input_tensor.to(device)
                expected_tensor = expected_tensor.to(device)
                output = model(input_tensor)

                output_data = extract_output(output)
                expected_data = extract_output(expected_tensor)

                scores[0] += 1/(abs(expected_data[0] - output_data[0]) + 1)
                scores[1] += 1/(abs(expected_data[1] - output_data[1]) + 1)
                scores[2] += 1/(abs(expected_data[2] - output_data[2]) + 1)
                scores[3] += 1 if expected_data[3] == output_data[3] else 0

                #print(f"Predicted: {extract_output(output)}")
                #print(f"Expected: {extract_output(expected_tensor)}")
                num_tested += 1

        print(output_data)
        print(expected_data)
        print("Done!")
        #[print(loss) for idx, loss in enumerate(losses) if not idx%50]
        [print(num_correct/num_tested) for num_correct in scores]
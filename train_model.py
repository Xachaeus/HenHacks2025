import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from tqdm import tqdm
import json
import math


def preprocess_instance(instance):

    # Outputs
    transaction_total = int(instance["valid total"])
    num_of_transactions = int(instance["num valid transactions"])
    could_continue = instance["could continue"]

    transaction_total_tensor = torch.tensor([int(bit) for bit in format(transaction_total & 0xFFFFFFFF, '032b')])
    num_of_transactions_tensor = torch.tensor([int(bit) for bit in format(num_of_transactions, '032b')])
    could_continue_tensor = torch.tensor([1.0 if could_continue else 0.0])

    #output_tensor = torch.cat((transaction_total_tensor, num_of_transactions_tensor, could_continue_tensor), dim=0)
    output_tensor = torch.tensor([float(transaction_total), float(num_of_transactions), float(could_continue)])
    # Inputs
    average_duration = int(instance["average duration"])
    category = float(instance["type id"])
    isHighschool = instance["Middle/High School"] == "High"
    launch_month = int(instance["launch month"])
    desired_operating_time = int(instance["valid duration"])

    average_duration_tensor = torch.tensor([int(bit) for bit in format(average_duration, '032b')])
    desired_operating_time_tensor = torch.tensor([int(bit) for bit in format(desired_operating_time, '032b')])
    #category_tensor = torch.tensor([(1.0 if category==c else 0.0) for c in categories])
    launch_month_tensor = torch.tensor([(1.0 if launch_month == num else 0.0) for num in range(1,13)])
    isHighschool_tensor = torch.tensor([1.0 if isHighschool else 0.0])

    #input_tensor = torch.cat((average_duration_tensor, desired_operating_time_tensor, category_tensor, launch_month_tensor, isHighschool_tensor), dim=0)
    input_tensor = torch.tensor([float(average_duration), float(desired_operating_time), float(category), float(isHighschool), float(launch_month), float(desired_operating_time)])

    return input_tensor, output_tensor


def extract_output(output_tensor):

    return (output_tensor[0].item(), output_tensor[1].item(), output_tensor[2].item()>0.5)

    data_tensor = torch.tensor([(1.0 if idx.item() > 0.5 else 0.0) for idx in output_tensor])
    output_arguments = [[int(x) for x in tensor.tolist()] for tensor in torch.tensor_split(data_tensor, [32,64], dim=0)]

    transaction_total = int("".join(map(str, output_arguments[0])), 2)
    num_transactions = int("".join(map(str, output_arguments[1])), 2)
    could_continue = output_arguments[2][0] == 1

    return (transaction_total, num_transactions, could_continue)
    



class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_ff):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, d_ff),
            nn.Linear(d_ff, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff,d_ff),
            nn.Linear(d_ff, output_dim),
        )

    def forward(self, x):
        return self.shared(x)
    



if __name__ == "__main__":

    AVERAGES = 1
    EPOCHS = 5
    LEARNING_RATE = 1e-5
    device = 'cuda'

    with open("preprocessed_dataset.json", 'r') as f: dataset = json.load(f)

    print(f"Dataset contains {len(dataset)} instances")

    train_dataset = [x for idx, x in enumerate(dataset) if idx%2]
    test_dataset = [x for idx, x in enumerate(dataset) if not idx%2]

    train_tensor_dataset = [preprocess_instance(instance) for instance in train_dataset]
    test_tensor_dataset = [preprocess_instance(instance) for instance in test_dataset]


    scores = [0,0,0]
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
                #output_data = extract_output(output)
                #expected_data = extract_output(output)
                #if(expected_data[2]==False and output_data[2]==True): loss *= 1000000
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        print("Done!\nBeginning testing...")
        # Test
        model.eval()
        with torch.no_grad():
            for iteration, (input_tensor, expected_tensor) in enumerate(test_tensor_dataset):
                input_tensor = input_tensor.to(device)
                expected_tensor = expected_tensor.to(device)
                output = model(input_tensor)

                output_data = extract_output(output)
                expected_data = extract_output(expected_tensor)

                scores[0] += 1/math.log((abs(expected_data[0] - output_data[0]) + 1))
                scores[1] += 1/math.log((abs(expected_data[1] - output_data[1]) + 1))
                scores[2] += 1 if expected_data[2] == output_data[2] else 0

                if not iteration % 50:
                    print(f"Predicted: {extract_output(output)}")
                    print(f"Expected: {extract_output(expected_tensor)}")
                num_tested += 1

        print(output_data)
        print(expected_data)
        print("Done!")
        #[print(loss) for idx, loss in enumerate(losses) if not idx%50]
        [print(num_correct/num_tested) for num_correct in scores]
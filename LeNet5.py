import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import warnings
from load_dataset import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LeNet5(nn.Module):

    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        # print(f"Output after layer2: {output.size()}")
        output = output.reshape(output.size(0), -1)
        output = self.fc(output)
        output = self.relu(output)
        output = self.fc1(output)
        output = self.relu1(output)
        output = self.fc2(output)
        return output


learning_rate = 0.001

model = LeNet5(num_classes).to(device)

criterion = nn.CrossEntropyLoss()

optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_steps = len(train_dataloader)

if __name__ == "__main__":
    print(f"Length of train_dataloader: {len(train_dataloader)}")
    print(f"Length of test_dataloader: {len(test_dataloader)}")

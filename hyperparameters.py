import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import warnings
import nvidia_smi
from LeNet5 import *
from load_dataset import *


def train(model, criterion, optimiser, num_epochs=10):
    total_training_loss = []
    total_steps = len(train_dataloader)

    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimiser.zero_grad()
            loss.backward()
            running_loss += loss.item() * images.size(0)
            optimiser.step()

            if (i + 1) % 400 == 0:
                print(f"Epoch [{epoch + 1} / {num_epochs}], Step [{i + 1} / {total_steps}, Loss {loss.item():.4f}]")

        epoch_loss = running_loss / len(train_dataloader)
        total_training_loss.append(epoch_loss)

    return total_training_loss

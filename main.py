import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import warnings
import nvidia_smi
from hyperparameters import *
from load_dataset import *
from LeNet5 import *

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    print(device)

total_training_loss = train(model, criterion, optimiser, num_epochs=30)

epoch_count = range(1, len(total_training_loss) + 1)

plt.plot(epoch_count, total_training_loss, "r--")
plt.legend(["Training Loss", "Test Loss"])
plt.xlabel("Epoch")
plt.show()

import torch
from train import CNN
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random

MODEL_PATH = "bin/mnist_cnn.pth"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5))
])

data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
test_loader = DataLoader(data, batch_size=100, shuffle=False)

# print(test_loader.shape)

model = CNN()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

@torch.no_grad()
def test_cnn(model, test_loader):
    idx = random.randint(0, len(test_loader.dataset) - 1)

    data, target = test_loader.dataset[idx]
    data = data.unsqueeze(0)

    image, label = test_loader.dataset[idx]
    image = image.squeeze(0).numpy()

    output = model(data)

    output = output.argmax(dim=1, keepdim=True)
    pred = output.item()

    plt.imshow(image, "gray")
    plt.title(f"Label: {label}, Prediction: {pred}")
    plt.show()

    print(f"Predicted: {pred}") 

test_cnn(model, test_loader)
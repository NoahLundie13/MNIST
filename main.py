import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5))
])

train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = DataLoader(test_data, batch_size=100, shuffle=False)

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()

    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # Color, num of feature maps, size of filter, padding to image pixels (1 means image stays same size)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

    self.pool = nn.MaxPool2d(2, 2)

    self.fc1 = nn.Linear(64 * 7 * 7, 128)  # input, output aka weight, bias?
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))  # Relu (activation function) adds non-linearity to the model
    x = self.pool(F.relu(self.conv2(x)))

    x = x.view(-1, 64 * 7 * 7)  # Flattens from 4d to 2d

    x = F.relu(self.fc1(x))
    x = self.fc2(x)  # Produces Logits -> Softmax function to turn into probabilities

    return x  # No softmax, CrossEntropyLoss expects logits


def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):  # Unpacks tuple returned from train loader into data, target variables
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()  

        output = model(data)  # Returns x

        loss = loss_fn(output, target)  # Applies softmax
        loss.backward()
        optimizer.step()  # Moves each weight by lr in whichever direction decreases loss

        running_loss += loss.item()  

        if batch_idx % 100 == 0:
            print(f"Train Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}") 

    print(f"Train Epoch Loss: {running_loss/len(train_loader):.4f}")  # avg loss, not necessary

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    accuracy = 100 * correct / total 
    print(f"Test Accuracy: {accuracy:.2f}%")

# --------------------------------------------------------------------------------------------------------------------------- #

model = CNN()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 10
for epoch in range(epochs): 
    print(f"\nEpoch {epoch+1}/{epochs}")
    train(model, train_loader, optimizer, loss_func, device)
    test(model, test_loader, device)

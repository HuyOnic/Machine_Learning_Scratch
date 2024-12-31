import torch
import torch.nn as nn
import json
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

class LeNet(torch.nn.Module):
    def __init__(self, num_class=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_class)

    def forward(self, x):
        x = self.avgpool(self.relu(self.conv1(x)))
        x = self.avgpool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)

if __name__ == "__main__":
    trf = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))])
    train_dataset = MNIST("./data", train=True, download=True, transform=trf)
    test_dataset = MNIST("./data", train=False, download=True, transform=trf)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    args = json.load(open("CNNBenchmark/config.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LeNet(10)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args["lr"], momentum=0.9, weight_decay=0.95)

    for epoch in range(1):
        running_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outs = model(images)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        running_loss /= len(train_loader)
        print(f"Epoch {epoch} Loss {running_loss}")
        torch.save(model.state_dict(), f"CNNBenchmark/exps/lenet_epoch{epoch}.pt")

    print("Evaluating Model")
    correct = 0
    num_samples = 0
    model.load_state_dict(torch.load("CNNBenchmark/exps/lenet_epoch0.pt"))
    model.eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outs = model(images)
            pred = torch.argmax(outs, dim=1)
            correct += (pred == labels).sum().item()
            num_samples += labels.size(0)
        print("Accuracy: ", correct / num_samples)
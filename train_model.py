import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split
from utils import get_transforms
import os

data_dir = r"C:\Users\HP\.vscode\cli\face-mask-detector\dataset"

if not os.path.exists(data_dir):
    raise Exception(f"Dataset path NOT found: {data_dir}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = get_transforms()

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
print("Classes found:", dataset.classes)

if len(dataset) == 0:
    raise Exception("No images found in dataset!")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = models.efficientnet_b0(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

for param in model.features[-4:].parameters():
    param.requires_grad = True

model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()

    train_acc = correct / train_size

    model.eval()
    val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()

    val_acc = val_correct / val_size

    print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

torch.save(model.state_dict(), "model.pth")

print("Model trained and saved!")

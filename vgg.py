import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.utils.prune as prune
import numpy as np
import random
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CIFAR10
from torch.utils.tensorboard import SummaryWriter

# Set random seed
random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)

writer = SummaryWriter("trial/v4-l1_norm/vgg")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
train_batch_size = 100
test_batch_size = 100
learning_rate = 0.001
num_epochs = 100
lambda_l1 = 0.0005
lambda_l2 = 0.0005
pruning_amount = 0.2  # Amount of pruning per layer

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR10 mean and std
])

train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)
valid_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)

# Define a simplified VGG model
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# Pruning functions
def apply_pruning_initialization(model, amount=pruning_amount):
    """Apply pruning to all Conv2D layers during initialization."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            print(f"Applied pruning to {name}")

def check_pruning(model):
    """Display the number of non-zero parameters for each pruned layer."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"{name} - Non-zero params: {module.weight.count_nonzero()} / Total: {module.weight.numel()}")

# Initialize model and apply pruning
model = VGG().to(device)
apply_pruning_initialization(model)

# Optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=lambda_l2)
criterion = nn.CrossEntropyLoss()

# Training function
def train(model, criterion, optimizer, train_loader, valid_loader, num_epochs):
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total

        valid_loss, valid_acc = evaluate(model, valid_loader, criterion)
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
              f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%")

        if valid_acc > best_accuracy:
            best_accuracy = valid_acc
            best_model_weights = model.state_dict()

    model.load_state_dict(best_model_weights)
    return model

# Evaluation function
def evaluate(model, valid_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(valid_loader.dataset)
    return avg_loss, accuracy

# Train and evaluate the model
trained_model = train(model, criterion, optimizer, train_loader, valid_loader, num_epochs)

# Check the pruning status after training
print("\nPruning status after training:")
check_pruning(trained_model)

writer.close()

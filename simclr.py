import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import numpy as np
import random
import time
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

# Set random seed
random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)

def main():
    writer = SummaryWriter("trial/v4-l1_norm/SimCLR")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    train_batch_size = 100
    test_batch_size = 100
    learning_rate = 0.001
    num_epochs = 100
    lambda_l1 = 0.0005
    lambda_l2 = 0.0005
    pruning_amount = 0.2  # Amount of pruning per layer

    # Data loading with SimCLR-style augmentation
    transform_simclr = transforms.Compose([
        transforms.RandomResizedCrop(size=28),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert to 3 channels
        transforms.ToTensor(),
        transforms.Normalize((0.1307,) * 3, (0.3081,) * 3)  # MNIST mean and std for 3 channels
    ])

    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform_simclr)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform_simclr)

    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    valid_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    # SimCLR Model Definition
    class SimCLR(nn.Module):
        def __init__(self, base_encoder, projection_dim=128):
            super(SimCLR, self).__init__()
            self.encoder = base_encoder(pretrained=False)
            self.encoder.fc = nn.Identity()  # Remove original fully connected layer

            # Add projection head
            self.projection_head = nn.Sequential(
                nn.Linear(512, 512),  # Adjust dimensions based on the encoder
                nn.ReLU(),
                nn.Linear(512, projection_dim)
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.projection_head(x)
            return x

    # Contrastive loss for SimCLR
    class NT_XentLoss(nn.Module):
        def __init__(self, batch_size, temperature=0.5):
            super(NT_XentLoss, self).__init__()
            self.batch_size = batch_size
            self.temperature = temperature
            self.criterion = nn.CrossEntropyLoss(reduction="sum")
            self.similarity_f = nn.CosineSimilarity(dim=2)

        def forward(self, z_i, z_j):
            N = 2 * self.batch_size
            z = torch.cat((z_i, z_j), dim=0)
            sim_matrix = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
            labels = torch.cat([torch.arange(self.batch_size) for _ in range(2)], dim=0)
            labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(z.device)
            mask = torch.eye(N, dtype=torch.bool).to(z.device)
            labels = labels[~mask].view(N, -1)
            sim_matrix = sim_matrix[~mask].view(N, -1)
            loss = self.criterion(sim_matrix, labels)
            return loss / N

    # Initialize model, loss, and optimizer
    model = SimCLR(base_encoder=torchvision.models.resnet18).to(device)
    contrastive_criterion = NT_XentLoss(batch_size=train_batch_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lambda_l2)

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

    # Apply initial pruning
    apply_pruning_initialization(model)

    # Training function with SimCLR contrastive learning
    def train_contrastive(model, contrastive_criterion, optimizer, train_loader, num_epochs):
        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for (x_i, _), (x_j, _) in zip(train_loader, train_loader):
                x_i, x_j = x_i.to(device), x_j.to(device)
                optimizer.zero_grad()
                
                z_i = model(x_i)
                z_j = model(x_j)
                
                loss = contrastive_criterion(z_i, z_j)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # Fine-tuning for MNIST classification
    def fine_tune_classification(model, train_loader, valid_loader, num_epochs):
        model.encoder.fc = nn.Linear(512, 10).to(device)  # Replace projection head for classification
        optimizer = optim.Adam(model.encoder.fc.parameters(), lr=learning_rate)

        criterion = nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = model.encoder(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

    # Execute pre-training and fine-tuning
    train_contrastive(model, contrastive_criterion, optimizer, train_loader, num_epochs=10)
    fine_tune_classification(model, train_loader, valid_loader, num_epochs=10)

    # Check pruning status after training
    print("\nPruning status after training:")
    check_pruning(model)

    writer.close()

# Entry point
if __name__ == '__main__':
    main()

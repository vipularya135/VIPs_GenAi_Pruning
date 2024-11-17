import profile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.utils.prune as prune
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.optim.lr_scheduler
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.utils.data import random_split
from torchvision.models import alexnet
from torch.utils.tensorboard import SummaryWriter
import time
import random

writer=SummaryWriter("trial/v4-l1_norm/first")
random_seed=42
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
train_batch_size = 100
test_batch_size=100
num_classes = 100
learning_rate = 0.001
num_epochs = 100
lambda_l1 = 0.0005
lambda_l2 = 0.0005
pruning_amount = 0.2

if torch.cuda.is_available():
   torch.cuda.manual_seed_all(random_seed)
   device=torch.device("cuda")
else:
   device=torch.device("cpu")

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # Mean and std deviation for MNIST grayscale images
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # Mean and std deviation for MNIST grayscale images
])   

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
test_size = len(test_dataset)

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)
valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)

from torchvision.models import mobilenet_v2

model = mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

def evaluate(model, valid_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss , _ = custom_loss(outputs, labels, model, criterion, lambda_l1, lambda_l2)  # Unpack the tuple
            total_loss += loss.item()  # Use loss.item() for accumulation
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * (correct / total)
    return total_loss, accuracy

def calculate_l1_norm_of_filters(model):
    l1_normalisation_values = {}
    l2_normalisation_values = {}

    for name, layer in model.named_children():
        if isinstance(layer, nn.Conv2d):
            filters = layer.weight
            l1_norm_of_filter = []
            l2_norm_of_filter = []

            for idx, filter in enumerate(filters):
                l1_norm = torch.sum(torch.abs(filter)).item()
                l2_norm = torch.sum(filter ** 2).item()
                l1_norm_of_filter.append(l1_norm)
                l2_norm_of_filter.append(l2_norm)

            l1_normalisation_values[name] = l1_norm_of_filter
            l2_normalisation_values[name] = l2_norm_of_filter
    print(l1_normalisation_values)
            # print(l2_normalisation_values)


    return l1_normalisation_values, l2_normalisation_values

# The function name remains same but it is calculating threshold according to both l1 and l2 norm
def calculate_threshold_l1_norm_of_filters(l1_normalisation_values, l2_normalisation_values, percentages_to_prune, lambda_l2):
    threshold_values = {}

    for layer, l1_norms in l1_normalisation_values.items():
        l2_norms = l2_normalisation_values[layer]
        thresholds = []

        for l1_norm, l2_norm in zip(l1_norms, l2_norms):
            threshold = l1_norm + lambda_l2 * l2_norm
            thresholds.append(threshold)

        sorted_thresholds = sorted(thresholds)
        percentage_to_prune = percentages_to_prune[layer]
        threshold_index = int(len(sorted_thresholds) * percentage_to_prune)
        threshold_value = sorted_thresholds[threshold_index]
        threshold_values[layer] = threshold_value

    print(threshold_values)
    return threshold_values


def index_remove(tensor, dim, index, removed=False):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    size_ = list(tensor.size())
    new_size = tensor.size(dim) - len(index)
    size_[dim] = new_size
    new_size = size_

    select_index = list(set(range(tensor.size(dim))) - set(index))
    select_index = torch.tensor(select_index, dtype=torch.int64)  # Cast indices to int64
    new_tensor = torch.index_select(tensor, dim, select_index)

    if removed:
        return new_tensor , torch.index_select(tensor, dim, torch.tensor(index))

    return new_tensor



def get_new_conv(in_channels,out_channels, conv, dim, channel_index, independent_prune_flag=False):
  # print(f"Doing for the layer {conv}")
  # print(f"Layer has {conv.in_channels} and should have {in_channels} as in channels")
  # print(f"Layer has {conv.out_channels} and should have {int(conv.out_channels - len(channel_index))} as in channels")

  new_conv = torch.nn.Conv2d(in_channels=in_channels,
                                   out_channels= out_channels,
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation)

  new_conv.weight.data = index_remove(conv.weight.data, 0, channel_index)
  # print(f"The layer {conv} has {len(new_conv.weight)} weights now ")
  new_conv.bias.data = index_remove(conv.bias.data, 0, channel_index)
  # print(f"The layer {conv} has {len(new_conv.bias)} bias now ")

  return new_conv

filters_to_remove=[]
def prune_filters(model, threshold_values):
    next_channel = 1
    conv_counter = 0  # Counter to track the number of Conv2d layers encountered
    for name, layer in model.named_children():
        filters_to_remove = []
        if isinstance(layer, nn.Conv2d):
            conv_counter += 1  # Increment the counter for each Conv2d layer
            filters = layer.weight
            num_filters_to_prune = 0

            for idx, filter in enumerate(filters):
                l1_norm = torch.sum(torch.abs(filter)).item()
                l2_norm = torch.sum(filter ** 2).item()
                threshold = l1_norm + lambda_l1 * l1_norm + lambda_l2 * l2_norm

                if l1_norm < threshold_values[name]:
                    num_filters_to_prune += 1
                    layer.weight.data[idx].zero_()
                    filters_to_remove.append(idx)

            if num_filters_to_prune > 0:
                in_channels = next_channel
                out_channels = layer.out_channels - num_filters_to_prune
                print(f"Filters to prune are {num_filters_to_prune}")
                print(f"The number of filters to prune are {filters_to_remove}")
                print(f"Making a new layer for {layer}")
                new_conv_layer = get_new_conv(in_channels, out_channels, layer, 0, filters_to_remove).to(device)
                setattr(model, name, new_conv_layer)
                next_channel = out_channels
                if conv_counter == 2:  # Check if it's the second Conv2d layer
                    print("Updating linear layer after the second Conv2d layer")
                    update_linear_layer(model, next_channel)  # Call the function to update the linear layer


        elif isinstance(layer, nn.BatchNorm2d):
            new_batch_norm_2d_layer = nn.BatchNorm2d(num_features=next_channel).to(device)
            setattr(model, name, new_batch_norm_2d_layer)
            del new_batch_norm_2d_layer

    if isinstance(layer, nn.Linear):

        in_features = next_channel * 4 * 4  # Adjust based on the size of the input after convolution layers
        out_features = layer.out_features
        has_bias = True if layer.bias is not None else False
        new_linear_layer = nn.Linear(in_features, out_features, bias=has_bias).to(device)
        setattr(model, name, new_linear_layer)
        del new_linear_layer

def update_linear_layer(model, next_channel):
    # Function to update the linear layer after the second Conv2d layer
    for name, layer in model.named_children():
        if isinstance(layer, nn.Linear):
            print("linear layer is updating ")
            in_features = next_channel * 4 * 4  # Adjust based on the size of the input after convolution layers
            out_features = layer.out_features
            has_bias = True if layer.bias is not None else False
            new_linear_layer = nn.Linear(in_features, out_features, bias=has_bias).to(device)
            setattr(model, name, new_linear_layer)
            del new_linear_layer
            break  # Exit loop after updating the linear layer


def check_pruning(model):
  print("\nLayer and filter sizes \n ------------------------------------")
  for name,module in model.named_modules():
    if isinstance(module,nn.Conv2d):
      print(f"Layer: {name}, Filter Size: {module.out_channels}")

# model = VGG16().to(device)
model = model().to(device)
print(f"Model is on device: {next(model.parameters()).device}")


percentages_to_prune = {
    'conv1': 0.1,
    'conv2': 0.15
}
threshold_values={}
l1_normalisation_values={}
l2_normalisation_values={}
l1_normalisation_values, l2_normalisation_values=calculate_l1_norm_of_filters(model)
threshold_values = calculate_threshold_l1_norm_of_filters(l1_normalisation_values, l2_normalisation_values, percentages_to_prune, lambda_l2)
# prune_filters(model,threshold_values)
# print("\n Pruned Filter Sizes \n")
# check_pruning(model)


optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=lambda_l2) 
criterion = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
total_step = len(train_loader)
loss_list = []
acc_list = []


epoch_list = []
train_loss_list = []
train_acc_list = []
valid_loss_list = []
valid_acc_list = []

best_model = None
best_accuracy = 0.0

pruning_epochs = 20
pruning_rate = 0.2

def update_inputs_channels(model):
  prev_channels=1
  for name,module in model.named_children():
    if isinstance(module,nn.Conv2d):
      in_channels=prev_channels
      module.weight.data = module.weight.data[:, :in_channels, :, :]
      # if module.bias is not None:
      #           module.bias.data = module.bias.data[:in_channels]
      module.in_channels=in_channels
      prev_channels=module.out_channels

def prune_model(model,pruning_rate):
   #l1_norm_values=calculate_l1_norm_of_filters(model)
   l1_normalisation_values, l2_normalisation_values=calculate_l1_norm_of_filters(model)
   threshold_values=threshold_values = calculate_threshold_l1_norm_of_filters(l1_normalisation_values, l2_normalisation_values, percentages_to_prune, lambda_l2)

   prune_filters(model,threshold_values)
   update_inputs_channels(model)

def print_remaining_filters(model):
   print("\nThe filters are \n -----------------------------------")
   for name,module in model.named_modules():
      if isinstance(module,nn.Conv2d):
         print(f"{name} has {module.out_channels} remaining filters")

def print_conv_layer_shapes(model):
    print("\nLayer and shape of the filters \n -----------------------------")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"Conv layer: {name}, Weight shape: {module.weight.shape}  Bias shape: {module.bias.shape if module.bias is not None else 'No bias'}")


print(torch.cuda.is_available())


print("\nBefore pruning:\n")
print_conv_layer_shapes(model)

prune_model(model,pruning_rate)

print("\nAfter pruning:\n")
print_conv_layer_shapes(model)

print("\n Pruned Filter Sizes \n")
check_pruning(model)


def custom_loss(outputs, labels, model, criterion, lambda_l1, lambda_l2):
    # Calculate the standard cross-entropy loss
    loss = criterion(outputs, labels)
    l12_lr= 0
    
    # Add L1 regularization
    l1_regularization = torch.tensor(0., device=device)
    for param in model.parameters():
        l1_regularization += torch.norm(param, 1)
    
    loss += lambda_l1 * l1_regularization
    l12_lr += lambda_l1 * l1_regularization
    
    # Add L2 regularization
    l2_regularization = torch.tensor(0., device=device)
    for param in model.parameters():
        l2_regularization += torch.norm(param, 2)**2
    
    loss += 0.5 * lambda_l2 * l2_regularization
    l12_lr += 0.5 * lambda_l2 * l2_regularization
    
    return loss, l12_lr

def print_loss_to_file(loss_list, filename):
    # Function to print loss values to a file
    with open(filename, 'w') as file:
        for epoch, loss in enumerate(loss_list, start=1):
            file.write(f'Epoch {epoch}: {loss}\n')
    print(f"Losses written to {filename}")
    

def train(model, criterion, optimizer, train_loader, valid_loader, num_epochs, lambda_l1, learning_rate, batch_size, weight_decay):
    best_model = None
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        l12_lr=0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            # Calculate custom loss
            loss, l12_lr = custom_loss(outputs, labels, model, criterion, lambda_l1, lambda_l2)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * (correct / total)

        valid_loss, valid_acc = evaluate(model, valid_loader, criterion)
        end_time = time.time()
        total_time = end_time - start_time
        ce = criterion(outputs, labels)

        print('\n  Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}%, L1,L2 Regularization: {:.2f}%, CrossEntropy Loss: {:.2f}%, Valid Loss: {:.4f}, Valid Accuracy: {:.2f}%, Time: {:.2f}s'.format(
            epoch + 1, num_epochs, epoch_loss, epoch_acc, l12_lr, ce, valid_loss, valid_acc, total_time))
        writer.add_scalar('Loss/Train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/Train', epoch_acc, epoch)
        writer.add_scalar('Loss/Valid', valid_loss, epoch)
        writer.add_scalar('Accuracy/Valid', valid_acc, epoch)
        if valid_acc > best_accuracy:
            best_accuracy = valid_acc
            best_model = model.state_dict()
        
        loss_list.append(epoch_loss)
    # Print loss to file after each epoch
    print_loss_to_file(loss_list, "loss_values3.txt")
    print("The best accuracy is ,",best_accuracy)

    return best_model

def update_model_after_pruning(model):
    prev_channels = 1  # Assuming input channels for the first layer is 1
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            module.in_channels = prev_channels
            prev_channels = module.out_channels

def train_model_with_best_hyperparameters(learning_rate, batch_size, num_epochs, weight_decay):
    # Define the model with hyperparameters
    # model = LeNet().to(device)
    print (model)
    update_model_after_pruning(model)
    # Define the optimizer with hyperparameters
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    
    # Define the criterion
    criterion = nn.CrossEntropyLoss()
    
    # Create data loaders with the specified batch size
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    update_model_after_pruning(model)
    # Train the model with the specified hyperparameters
    best_model_weights = train(model, criterion, optimizer, train_loader, valid_loader, num_epochs, lambda_l1, learning_rate, batch_size, weight_decay)
    
    # Load the best model weights
    model.load_state_dict(best_model_weights)
    
    return model

batch_size =100
weight_decay=5e-4
for epoch in range (pruning_epochs) :
    model = train_model_with_best_hyperparameters(learning_rate, batch_size, num_epochs, weight_decay)
    
    print("\nBefore pruning:\n")
    print_conv_layer_shapes(model)

    prune_model(model,pruning_rate)

    print("\nAfter pruning:\n")
    print_conv_layer_shapes(model)

    print("\n Pruned Filter Sizes \n")
    check_pruning(model)

writer.close()
for name,module in model.named_children():
  if isinstance(module,nn.Conv2d):
    print(f"\nBias of layer {module} is {len(module.bias.data)}")

print_conv_layer_shapes(model)
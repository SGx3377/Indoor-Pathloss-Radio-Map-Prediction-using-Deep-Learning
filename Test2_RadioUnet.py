import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.pyplot as plt
from skimage.io import imread
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchsummary import summary
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Define the dataset class
class RadioMapDataset(Dataset):
    def __init__(self, input_dir, output_dir, buildings, input_transform=None, output_transform=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.samples = sorted([f.split('_S')[0] + '_S' + f.split('_S')[1].split('.')[0]
                               for f in os.listdir(input_dir)
                               if f.split('_')[0] in buildings])
        self.epsilon = 1e-6  # Small value to avoid division by zero

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_image_path = os.path.join(self.input_dir, self.samples[idx] + '.png')
        output_image_path = os.path.join(self.output_dir, self.samples[idx] + '.png')

        input_image = Image.open(input_image_path).convert('RGB')
        output_image = Image.open(output_image_path).convert('L')  # Grayscale

        # Store original size for later use
        original_size = input_image.size

        # Apply transforms if provided
        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.output_transform:
            output_image = self.output_transform(output_image)

        # Process input image to add the fourth channel
        first_channel = input_image[0, :, :]
        second_channel = input_image[1, :, :]
        third_channel = input_image[2, :, :]

        fourth_channel = -(first_channel + second_channel) / (third_channel + self.epsilon)
        fourth_channel = (fourth_channel - fourth_channel.min()) / (fourth_channel.max() - fourth_channel.min())
        fourth_channel = fourth_channel.unsqueeze(0)  # Add a channel dimension

        input_image = torch.cat((input_image, fourth_channel), dim=0)
        
         # Calculate min and max values for input data
        input_min, input_max = input_image.min().item(), input_image.max().item()
        # print(f"Input Data Range - Min: {input_min}, Max: {input_max}")

        return input_image, output_image, original_size


# Transform functions
def synchronized_transform(img, rotation_angle,flip):
    if flip == Image.FLIP_LEFT_RIGHT:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip == Image.FLIP_TOP_BOTTOM:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img = img.rotate(rotation_angle,expand=True)
    img = img.resize((256, 256))
    return img


def combined_transform(img, rotation_angle,flip):
    if img.mode == "RGB":
        img = synchronized_transform(img, rotation_angle,flip)
    elif img.mode == "L":
        img = synchronized_transform(img, rotation_angle,flip)
    return img


def get_train_transform():
    rotation_angle = random.choice([0, 90, 180, 270])
    flip = random.choice([Image.FLIP_LEFT_RIGHT,Image.FLIP_TOP_BOTTOM])
    input_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        transforms.Lambda(lambda img: combined_transform(img, rotation_angle,flip)),
        transforms.ToTensor()
    ])
    output_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("L") if img.mode != "L" else img),
        transforms.Lambda(lambda img: combined_transform(img, rotation_angle,flip)),
        transforms.ToTensor()
    ])
    return input_transform, output_transform


input_transform, output_transform = get_train_transform()

val_test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])



# Define buildings
all_buildings = [f"B{i}" for i in range(1, 26)]
print(all_buildings)

np.random.seed(0)
np.random.shuffle(all_buildings)

# Split into train, validation, and test sets based on ranges
buildings_train = all_buildings[:17]  # First 17 buildings
buildings_val = all_buildings[17:21]  # Next 4 buildings
buildings_test = all_buildings[21:]   # Remaining 4 buildingsi

# Data directories
train_input_dir = 'ICASSP2025_Dataset/Inputs/Task_1_ICASSP_Augmented_Inputs'
train_output_dir = 'ICASSP2025_Dataset/Outputs/Task_1_ICASSP_Augmented_Outputs'

# Datasets and loaders
train_dataset = RadioMapDataset(train_input_dir, train_output_dir, buildings_train, input_transform=input_transform, output_transform=output_transform)
val_dataset = RadioMapDataset(train_input_dir, train_output_dir, buildings_val, input_transform=val_test_transform, output_transform=val_test_transform)
test_dataset = RadioMapDataset(train_input_dir, train_output_dir, buildings_test, input_transform=val_test_transform, output_transform=val_test_transform)


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Unet Model- According to the RadioUnet architecture
def convrelu(in_channels, out_channels, kernel, padding, pool):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool, stride=pool, padding=0)
    )


def convreluT(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=2, padding=padding),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.layer00 = convrelu(in_channels, 10, 3, 1, 1)
        self.layer0 = convrelu(10, 40, 5, 2, 2)
        self.layer1 = convrelu(40, 50, 5, 2, 2)
        self.layer10 = convrelu(50, 60, 5, 2, 1)
        self.layer2 = convrelu(60, 100, 5, 2, 2)
        self.layer20 = convrelu(100, 100, 3, 1, 1)
        self.layer3 = convrelu(100, 150, 5, 2, 2)
        self.layer4 = convrelu(150, 300, 5, 2, 2)
        self.layer5 = convrelu(300, 500, 5, 2, 2)
        
        self.conv_up5 = convreluT(500, 300, 4, 1)
        self.conv_up4 = convreluT(300 + 300, 150, 4, 1)
        self.conv_up3 = convreluT(150 + 150, 100, 4, 1)
        self.conv_up20 = convrelu(100 + 100, 100, 3, 1, 1)
        self.conv_up2 = convreluT(100 + 100, 60, 6, 2)
        self.conv_up10 = convrelu(60 + 60, 50, 5, 2, 1)
        self.conv_up1 = convreluT(50 + 50, 40, 6, 2)
        self.conv_up0 = convreluT(40 + 40, 20, 6, 2)
        self.conv_up00 = convrelu(20 + 10 + in_channels, 20, 5, 2, 1)
        self.conv_up000 = convrelu(20 + in_channels, out_channels, 5, 2, 1)

        

    def forward(self, input):
        input0 = input[:, :self.in_channels, :, :]
        
        layer00 = self.layer00(input0)
        layer0 = self.layer0(layer00)
        layer1 = self.layer1(layer0)
        layer10 = self.layer10(layer1)
        layer2 = self.layer2(layer10)
        layer20 = self.layer20(layer2)
        layer3 = self.layer3(layer20)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
    
        layer4u = self.conv_up5(layer5)
        layer4u = torch.cat([layer4u, layer4], dim=1)
        layer3u = self.conv_up4(layer4u)
        layer3u = torch.cat([layer3u, layer3], dim=1)
        layer20u = self.conv_up3(layer3u)
        layer20u = torch.cat([layer20u, layer20], dim=1)
        layer2u = self.conv_up20(layer20u)
        layer2u = torch.cat([layer2u, layer2], dim=1)
        layer10u = self.conv_up2(layer2u)
        layer10u = torch.cat([layer10u, layer10], dim=1)
        layer1u = self.conv_up10(layer10u)
        layer1u = torch.cat([layer1u, layer1], dim=1)
        layer0u = self.conv_up1(layer1u)
        layer0u = torch.cat([layer0u, layer0], dim=1)
        layer00u = self.conv_up0(layer0u)
        layer00u = torch.cat([layer00u, layer00], dim=1)
        layer00u = torch.cat([layer00u, input0], dim=1)
        layer000u = self.conv_up00(layer00u)
        layer000u = torch.cat([layer000u, input0], dim=1)
        output = self.conv_up000(layer000u)
        
        return output

# Initialize model
model = UNet(in_channels=4, out_channels=1).cuda()

# Define your device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Model summary
summary(model, (4, 256, 256))

# Lists to store loss values
train_losses = []


# Define loss function with resizing each image back to its original size
def compute_loss(model_output, target, original_sizes, criterion):
    total_loss = 0.0
    batch_size = model_output.size(0)

    for i in range(batch_size):
        # Extract height and width for resizing from original_sizes
        height = original_sizes[0][i].item()
        width = original_sizes[1][i].item()
        original_size = (height, width)

        # Resize model output and target to the original size of each image
        model_output_resized = F.interpolate(model_output[i:i+1], size=original_size, mode='bilinear', align_corners=True)
        target_resized = F.interpolate(target[i:i+1], size=original_size, mode='bilinear', align_corners=True)

        # Compute the loss for this image
        loss = criterion(model_output_resized, target_resized)
        total_loss += loss

    return total_loss / batch_size

# Validation function
def validate_model(model, val_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets, original_sizes in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = compute_loss(outputs, targets, original_sizes, criterion)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss

# Testing function
def test_model(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets, original_sizes in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = compute_loss(outputs, targets, original_sizes, criterion)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)
    print(f'Final Test Loss: {avg_test_loss:.5f}')
    return avg_test_loss

# Training Function with Early Stopping
class EarlyStopping:
    def __init__(self, patience=50):
        self.patience = patience
        self.best_loss = float('inf')
        self.best_epoch = 0  

    def __call__(self, val_loss, current_epoch):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = current_epoch  # Update best epoch
            return False  

        return (current_epoch - self.best_epoch) >= self.patience

# Training function with early stopping and loss curve saving
def train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, num_epochs=200, checkpoint_dir="checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    early_stopping = EarlyStopping(patience=50)
    best_val_loss = float('inf')
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets, original_sizes in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = validate_model(model, val_loader, criterion)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.5f}, Validation Loss: {avg_val_loss:.5f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            checkpoint_path = os.path.join(checkpoint_dir, f'best_model_epoch_{best_epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved at {checkpoint_path}')

        if early_stopping(avg_val_loss, epoch + 1):  # Pass the current epoch
            print("Early stopping triggered")
            break

    loss_plot_path = os.path.join(checkpoint_dir, "train_val_loss_plot.png")
    train_loss_plot_path = os.path.join(checkpoint_dir, "train_loss_plot.png")
    val_loss_plot_path = os.path.join(checkpoint_dir, "val_loss_plot.png")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss Curves')
    plt.legend()
    plt.grid()
    plt.savefig(loss_plot_path)  # Save combined plot
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid()
    plt.savefig(train_loss_plot_path)  # Save train loss plot
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss Curve')
    plt.legend()
    plt.grid()
    plt.savefig(val_loss_plot_path)  # Save validation loss plot
    plt.close()

    print(f"Loss plots saved: {loss_plot_path}, {train_loss_plot_path}, {val_loss_plot_path}")


# Example usage
num_epochs = 200
train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)

# Find the latest checkpoint
checkpoint_dir = "checkpoints"

# Create the directory if it does not exist
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("best_model_epoch_") and f.endswith(".pth")]

if not checkpoint_files:
    raise FileNotFoundError("No checkpoint files found in 'checkpoints' directory.")

# Extract epoch numbers and find the highest one
latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))

# Construct the full path to the best checkpoint
best_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)


print(f"Loading checkpoint: {best_checkpoint_path}")

# Load the checkpoint safely
model.load_state_dict(torch.load(best_checkpoint_path, weights_only=True))

# Test the model
test_model(model, test_loader, criterion)
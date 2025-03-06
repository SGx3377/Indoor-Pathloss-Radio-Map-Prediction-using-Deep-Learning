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
        print(f"Input Data Range - Min: {input_min}, Max: {input_max}")

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
buildings_test = all_buildings[21:]   # Remaining 4 buildings

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



# ASPP module (Atrous Spatial Pyramid Pooling)
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1))

        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(x))
        conv3 = self.relu(self.conv3(x))
        conv4 = self.relu(self.conv4(x))
        global_avg_pool = self.global_avg_pool(x)
        global_avg_pool = F.interpolate(global_avg_pool, size=conv1.size()[2:], mode='bilinear', align_corners=True)

        out = torch.cat([conv1, conv2, conv3, conv4, global_avg_pool], dim=1)
        out = self.conv_out(out)
        return out


# U-Net model with ASPP
class UNetASPP(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(UNetASPP, self).__init__()

        def conv_block(in_channels, out_channels):
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            return block

        def up_conv(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # Encoder
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        # ASPP before bottleneck
        self.aspp = ASPP(512, 1024)

        # Bottleneck
        self.bottleneck = conv_block(1024, 1024)

        # Decoder
        self.upconv4 = up_conv(1024, 512)
        self.dec4 = conv_block(1024, 512)
        self.upconv3 = up_conv(512, 256)
        self.dec3 = conv_block(512, 256)
        self.upconv2 = up_conv(256, 128)
        self.dec2 = conv_block(256, 128)
        self.upconv1 = up_conv(128, 64)
        self.dec1 = conv_block(128, 64)

        # Output
        self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)


    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # ASPP + Bottleneck
        aspp_out = self.aspp(enc4)
        bottleneck = self.bottleneck(F.max_pool2d(aspp_out, 2))

        # Decoder path with resizing
        up4 = F.interpolate(self.upconv4(bottleneck), size=enc4.size()[2:], mode='bilinear', align_corners=True)
        dec4 = self.dec4(torch.cat((up4, enc4), dim=1))

        up3 = F.interpolate(self.upconv3(dec4), size=enc3.size()[2:], mode='bilinear', align_corners=True)
        dec3 = self.dec3(torch.cat((up3, enc3), dim=1))

        up2 = F.interpolate(self.upconv2(dec3), size=enc2.size()[2:], mode='bilinear', align_corners=True)
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))

        up1 = F.interpolate(self.upconv1(dec2), size=enc1.size()[2:], mode='bilinear', align_corners=True)
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))

        # Output layer
        out = self.conv_last(dec1)
        return out



# Initialize the model
model = UNetASPP(in_channels=4, out_channels=1).cuda()

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
    print(f'Final Test Loss: {avg_test_loss:.4f}')
    return avg_test_loss

# Training function with validation and checkpointing
def train_model_with_validation_and_checkpointing(model, train_loader, val_loader, criterion, optimizer, num_epochs=1, checkpoint_dir="checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    best_val_loss = float('inf')

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, targets, original_sizes) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = compute_loss(outputs, targets, original_sizes, criterion)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}] completed with Average Training Loss: {avg_train_loss:.5f}')

        # Perform validation
        avg_val_loss = validate_model(model, val_loader, criterion)
        print(f'Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_val_loss:.5f}')

        # Checkpoint if validation loss has improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved at {checkpoint_path} with Validation Loss: {best_val_loss:.5f}')

# Example usage
num_epochs = 150
# Initialize model, criterion, optimizer, and dataloaders (train_loader, val_loader, test_loader) as per your setup
train_model_with_validation_and_checkpointing(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)

# Load the best checkpoint for testing
best_checkpoint_path = os.path.join("T1_new_checkpoints", "best_model_epoch.pth")
model.load_state_dict(torch.load(best_checkpoint_path))

# Test the model
avg_test_loss = test_model(model, test_loader, criterion)
print(f'Final Test Loss: {avg_test_loss:.4f}')



# Initialize your model
# checkpoint = torch.load('checkpoints/best_model.pth', weights_only=True)  # Load checkpoint
# model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # Load only model weights



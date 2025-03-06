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
from tqdm import tqdm
import wandb


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()
wandb.login()
wandb.init(project="PMNET",name="PMNET_Training_2")

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

        # # Process input image to add the fourth channel
        # first_channel = input_image[0, :, :]
        # second_channel = input_image[1, :, :]
        # third_channel = input_image[2, :, :]

        # fourth_channel = -(first_channel + second_channel) / (third_channel + self.epsilon)
        # fourth_channel = (fourth_channel - fourth_channel.min()) / (fourth_channel.max() - fourth_channel.min())
        # fourth_channel = fourth_channel.unsqueeze(0)  # Add a channel dimension

        # input_image = torch.cat((input_image, fourth_channel), dim=0)
        
        #  # Calculate min and max values for input data
        # input_min, input_max = input_image.min().item(), input_image.max().item()
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
train_input_dir = '../ICASSP2025_Dataset/Inputs/Task_1_ICASSP_Augmented_Inputs'
train_output_dir = '../ICASSP2025_Dataset/Outputs/Task_1_ICASSP_Augmented_Outputs'

# Datasets and loaders
train_dataset = RadioMapDataset(train_input_dir, train_output_dir, buildings_train, input_transform=input_transform, output_transform=output_transform)
val_dataset = RadioMapDataset(train_input_dir, train_output_dir, buildings_val, input_transform=val_test_transform, output_transform=val_test_transform)
test_dataset = RadioMapDataset(train_input_dir, train_output_dir, buildings_test, input_transform=val_test_transform, output_transform=val_test_transform)


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

_BATCH_NORM = nn.BatchNorm2d

_BOTTLENECK_EXPANSION = 4

# Conv, Batchnorm, Relu layers, basic building block.
class _ConvBnReLU(nn.Sequential):

    BATCH_NORM = _BATCH_NORM

    def __init__(
        self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", _BATCH_NORM(out_ch, eps=1e-5, momentum=1 - 0.999))

        if relu:
            self.add_module("relu", nn.ReLU())

# Bottleneck layer cinstructed from ConvBnRelu layer block, buiding block for Res layers
class _Bottleneck(nn.Module):

    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        mid_ch = out_ch // _BOTTLENECK_EXPANSION
        self.reduce = _ConvBnReLU(in_ch, mid_ch, 1, stride, 0, 1, True)
        self.conv3x3 = _ConvBnReLU(mid_ch, mid_ch, 3, 1, dilation, dilation, True)
        self.increase = _ConvBnReLU(mid_ch, out_ch, 1, 1, 0, 1, False)
        self.shortcut = (
            _ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, False)
            if downsample
            else nn.Identity()
        )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)

# Res Layer used to costruct the encoder
class _ResLayer(nn.Sequential):

    def __init__(self, n_layers, in_ch, out_ch, stride, dilation, multi_grids=None):
        super(_ResLayer, self).__init__()

        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(multi_grids)

        # Downsampling is only in the first block
        for i in range(n_layers):
            self.add_module(
                "block{}".format(i + 1),
                _Bottleneck(
                    in_ch=(in_ch if i == 0 else out_ch),
                    out_ch=out_ch,
                    stride=(stride if i == 0 else 1),
                    dilation=dilation * multi_grids[i],
                    downsample=(True if i == 0 else False),
                ),
            )

# Stem layer is the initial interfacing layer
class _Stem(nn.Sequential):
    """
    The 1st conv layer.
    Note that the max pooling is different from both MSRA and FAIR ResNet.
    """

    def __init__(self, out_ch, in_ch=3):
        super(_Stem, self).__init__()
        self.add_module("conv1", _ConvBnReLU(in_ch, out_ch, 7, 2, 3, 1))
        self.add_module("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True))

class _ImagePool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        h = self.pool(x)
        h = self.conv(h)
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        return h


# Atrous spatial pyramid pooling
class _ASPP(nn.Module):

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module("c0", _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1))
        for i, rate in enumerate(rates):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBnReLU(in_ch, out_ch, 3, 1, padding=rate, dilation=rate),
            )
        self.stages.add_module("imagepool", _ImagePool(in_ch, out_ch))

    def forward(self, x):
        return torch.cat([stage(x) for stage in self.stages.children()], dim=1)


# Decoder layer constricted using these 2 blocks
def ConRu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True)
    )

def ConRuT(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=2, padding=padding),
        nn.ReLU(inplace=True)
    )

class PMNet(nn.Module):
    def __init__(self, n_blocks, atrous_rates, multi_grids, output_stride):
        super(PMNet, self).__init__()
        # Encoder (unchanged)
        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]
        ch = [64 * 2 ** p for p in range(6)] 
        self.layer1 = _Stem(ch[0])
        self.layer2 = _ResLayer(n_blocks[0], ch[0], ch[2], s[0], d[0])
        self.layer3 = _ResLayer(n_blocks[1], ch[2], ch[3], s[1], d[1])
        self.layer4 = _ResLayer(n_blocks[2], ch[3], ch[3], s[2], d[2])
        self.layer5 = _ResLayer(n_blocks[3], ch[3], ch[4], s[3], d[3], multi_grids)
        self.aspp = _ASPP(ch[4], 256, atrous_rates)
        concat_ch = 256 * (len(atrous_rates) + 2)
        self.add_module("fc1", _ConvBnReLU(concat_ch, 512, 1, 1, 0, 1))
        self.reduce = _ConvBnReLU(256, 256, 1, 1, 0, 1)

        # Decoder with Dropout
        self.conv_up5 = ConRu(512, 512, 3, 1)
        self.conv_up4 = ConRu(512 + 512, 512, 3, 1)
        self.conv_up3 = ConRuT(512 + 512, 256, 3, 1)
        self.conv_up2 = ConRu(256 + 256, 256, 3, 1)
        self.conv_up1 = ConRu(256 + 256, 256, 3, 1)

        self.conv_up0 = ConRu(256 + 64, 128, 3, 1)
        self.conv_up00 = nn.Sequential(
            nn.Conv2d(128 + 3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Encoder (unchanged)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.reduce(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.layer5(x5)
        x7 = self.aspp(x6)
        x8 = self.fc1(x7)

        # Decoder with Dropout
        xup5 = self.conv_up5(x8)
        xup5 = torch.cat([xup5, x5], dim=1)

        xup4 = self.conv_up4(xup5)
        xup4 = torch.cat([xup4, x4], dim=1)

        xup3 = self.conv_up3(xup4)
        xup3 = F.interpolate(xup3, size=x3.shape[2:], mode="bilinear", align_corners=False)
        xup3 = torch.cat([xup3, x3], dim=1)

        xup2 = self.conv_up2(xup3)
        xup2 = torch.cat([xup2, x2], dim=1)

        xup1 = self.conv_up1(xup2)
        xup1 = torch.cat([xup1, x1], dim=1)

        xup0 = self.conv_up0(xup1)
        xup0 = F.interpolate(xup0, size=x.shape[2:], mode="bilinear", align_corners=False)
        xup0 = torch.cat([xup0, x], dim=1)

        xup00 = self.conv_up00(xup0)
        return xup00
    
# Initialize model
model = PMNet(
            n_blocks=[3, 3, 27, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=8,).cuda()

# Define your device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Model summary
summary(model, (3, 256, 256))

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
        for inputs, targets, original_sizes in tqdm(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = compute_loss(outputs, targets, original_sizes, criterion)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)
    wandb.log({"Test Loss": avg_test_loss, })
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
    
writer = SummaryWriter(log_dir="runs/pmnet_experiment_1")
dummy_input = torch.randn(1, 3, 256, 256).to(device)  # Adjust input size as needed
writer.add_graph(model, dummy_input)

# Training function with early stopping and loss curve saving
def train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, num_epochs=200, checkpoint_dir="checkpoints_pmnet"):
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

        for inputs, targets, original_sizes in tqdm(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = validate_model(model, val_loader, criterion)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        wandb.log({"Train Loss": avg_train_loss,"Validation Loss": avg_val_loss, })

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
checkpoint_dir = "checkpoints_pmnet"

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

wandb.finish()
writer.close()
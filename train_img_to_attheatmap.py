import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import glob
import os
import cfg_reverse_adaptation
# import function_r as function
import matplotlib.pyplot as plt

class SmallUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, init_features=16):  # Start with fewer initial features
        super(SmallUNet, self).__init__()

        features = init_features
        self.encoder1 = SmallUNet._block(in_channels, features, name="enc1")
        self.encoder2 = SmallUNet._block(features, features * 2, name="enc2")
        self.encoder3 = SmallUNet._block(features * 2, features * 4, name="enc3")
        self.encoder4 = SmallUNet._block(features * 4, features * 8, name="enc4")

        self.bottleneck = SmallUNet._block(features * 8, features * 8, name="bottleneck")

        self.decoder4 = SmallUNet._block(features * 8 + features * 8, features * 4, name="dec4")
        self.decoder3 = SmallUNet._block(features * 4 + features * 4, features * 2, name="dec3")
        self.decoder2 = SmallUNet._block(features * 2 + features * 2, features, name="dec2")
        self.decoder1 = SmallUNet._block(features + features, features, name="dec1")

        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(features * 8, features * 8, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(features * 4, features * 4, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(features * 2, features * 2, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(features, features, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)  # Concatenate with encoder output
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )



class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Example data paths (replace with your actual paths)
# image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", ...]
# mask_paths = ["path/to/mask1.jpg", "path/to/mask2.jpg", ...]
image_directory = "dataset/TestDataset/CVC-ClinicDB/images"
saliency_directory = "dataset/TestDataset/CVC-ClinicDB_atta_heatmap"
# Use glob to get all image paths with specified extensions
image_paths = glob.glob(os.path.join(image_directory, "*.[jp][pn]*g"))
saliency_path = glob.glob(os.path.join(saliency_directory, "*.[jp][pn]*g"))
#
# # Print all image paths (optional)
# for path in image_paths:
#     print(path)
# for path in saliency_path:
#     print(path)

# Split into train and validation sets
train_images, val_images, train_masks, val_masks = train_test_split(image_paths, saliency_path, test_size=0.1)

args = cfg_reverse_adaptation.parse_args()

transform_train = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
])

transform_train_seg = transforms.Compose([
    transforms.Resize((args.out_size, args.out_size)),
    transforms.ToTensor(),
])


# Create datasets and data loaders
train_dataset = CustomDataset(train_images, train_masks, transform=transform_train)
val_dataset = CustomDataset(val_images, val_masks, transform=transform_train)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallUNet(in_channels=3, out_channels=3).to(device)  # Assuming grayscale images and masks
criterion = nn.BCELoss()  # Binary cross-entropy for binary segmentation or saliency maps
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training parameters
num_epochs = 20
best_val_loss = float("inf")

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Save the model with the best validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "checkpoint/best_unet_model.pth")
        print("Model saved!")

print("Training completed.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallUNet(in_channels=3, out_channels=3, init_features=16).to(device)
model.load_state_dict(torch.load("checkpoint/best_unet_model.pth", map_location=device))
model.eval()

inverse_transform = transforms.Compose([
    transforms.ToPILImage()
])

input_dir = "dataset/TestDataset/CVC-ClinicDB/images"
output_dir = "evalDataset/save_predictions"
os.makedirs(output_dir, exist_ok=True)

# Loop through each image in the test directory
for filename in os.listdir(input_dir):
    if filename.endswith((".jpg", ".png", ".jpeg")):  # Adjust extensions as needed
        # Load and preprocess the image
        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform_train(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)

        output = output.squeeze(0).cpu()  # Remove batch dimension and move to CPU
        output_image = inverse_transform(output)  # Convert tensor to PIL image

        # Save the prediction
        output_image.save(os.path.join(output_dir, f"pred_{filename}"))
        print(f"Saved prediction for {filename} as pred_{filename}")

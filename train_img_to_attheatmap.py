import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import glob
import os
import cfg_reverse_adaptation
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image


# class SmallUNet(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3, init_features=16):
#         super(SmallUNet, self).__init__()
#
#         features = init_features
#         self.encoder1 = SmallUNet._block(in_channels, features, name="enc1")
#         self.encoder2 = SmallUNet._block(features, features * 2, name="enc2")
#         self.encoder3 = SmallUNet._block(features * 2, features * 4, name="enc3")
#         self.encoder4 = SmallUNet._block(features * 4, features * 8, name="enc4")
#
#         self.bottleneck = SmallUNet._block(features * 8, features * 8, name="bottleneck")
#
#         self.decoder4 = SmallUNet._block(features * 8 + features * 8, features * 4, name="dec4")
#         self.decoder3 = SmallUNet._block(features * 4 + features * 4, features * 2, name="dec3")
#         self.decoder2 = SmallUNet._block(features * 2 + features * 2, features, name="dec2")
#         self.decoder1 = SmallUNet._block(features + features, features, name="dec1")
#
#         self.conv = nn.Conv2d(features, out_channels, kernel_size=1)
#
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.upconv4 = nn.ConvTranspose2d(features * 8, features * 8, kernel_size=2, stride=2)
#         self.upconv3 = nn.ConvTranspose2d(features * 4, features * 4, kernel_size=2, stride=2)
#         self.upconv2 = nn.ConvTranspose2d(features * 2, features * 2, kernel_size=2, stride=2)
#         self.upconv1 = nn.ConvTranspose2d(features, features, kernel_size=2, stride=2)
#
#     def forward(self, x):
#         enc1 = self.encoder1(x)
#         enc2 = self.encoder2(self.pool(enc1))
#         enc3 = self.encoder3(self.pool(enc2))
#         enc4 = self.encoder4(self.pool(enc3))
#
#         bottleneck = self.bottleneck(self.pool(enc4))
#
#         dec4 = self.upconv4(bottleneck)
#         dec4 = torch.cat((dec4, enc4), dim=1)
#         dec4 = self.decoder4(dec4)
#
#         dec3 = self.upconv3(dec4)
#         dec3 = torch.cat((dec3, enc3), dim=1)
#         dec3 = self.decoder3(dec3)
#
#         dec2 = self.upconv2(dec3)
#         dec2 = torch.cat((dec2, enc2), dim=1)
#         dec2 = self.decoder2(dec2)
#
#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat((dec1, enc1), dim=1)
#         dec1 = self.decoder1(dec1)
#
#         return self.conv(dec1)
#     @staticmethod
#     def _block(in_channels, features, name):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(features),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(features),
#             nn.ReLU(inplace=True)
#         )
#
# class Discriminator(nn.Module):
#     def __init__(self, in_channels=6):  # Concatenated input and target images
#         super(Discriminator, self).__init__()
#         def discriminator_block(in_filters, out_filters, normalization=True):
#             layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
#             if normalization:
#                 layers.append(nn.InstanceNorm2d(out_filters))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers
#
#         self.model = nn.Sequential(
#             *discriminator_block(in_channels, 64, normalization=False),
#             *discriminator_block(64, 128),
#             *discriminator_block(128, 256),
#             *discriminator_block(256, 512),
#             nn.Conv2d(512, 1, kernel_size=4, padding=1)
#         )
#
#     def forward(self, img_input, img_target):
#         # Concatenate input and target images
#         x = torch.cat((img_input, img_target), dim=1)
#         return self.model(x)
#     @staticmethod
#     def _block(in_channels, features, name):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(features),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(features),
#             nn.ReLU(inplace=True)
#         )
#
class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform_image=None, transform_mask=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform_image = transform_image
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("RGB")
#
#         if self.transform_image:
#             image = self.transform_image(image)
#         if self.transform_mask:
#             mask = self.transform_mask(mask)
#
#         return image, mask
#
# # Load image and mask paths
image_directory = "dataset/TestDataset/CVC-ClinicDB/images"
saliency_directory = "dataset/TestDataset/CVC-ClinicDB_atta_heatmap"

# Get all image paths
image_paths = glob.glob(os.path.join(image_directory, "*.[jp][pn]*g"))
saliency_paths = glob.glob(os.path.join(saliency_directory, "*.[jp][pn]*g"))
#
# # Sort the lists to ensure correct pairing
image_paths.sort()
saliency_paths.sort()
#
# Check if the number of images and masks are equal
if len(image_paths) != len(saliency_paths):
    raise ValueError(f"Number of images ({len(image_paths)}) and masks ({len(saliency_paths)}) do not match.")

# Split into train and validation sets
train_images, val_images, train_masks, val_masks = train_test_split(
    image_paths, saliency_paths, test_size=0.0, random_state=42
)

args = cfg_reverse_adaptation.parse_args()
#
# # Define transformations
# transform_image = transforms.Compose([
#     transforms.Resize((args.image_size, args.image_size)),
#     transforms.ToTensor(),
# ])
#
# transform_mask = transforms.Compose([
#     transforms.Resize((args.out_size, args.out_size)),
#     transforms.ToTensor(),
# ])
transform_image = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to a fixed size
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize pixel values
])

# # Create datasets and data loaders
train_dataset = CustomDataset(train_images, train_masks, transform_image=transform_image, transform_mask=transform_image)
val_dataset = CustomDataset(val_images, val_masks, transform_image=transform_image, transform_mask=transform_image)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
#
# def train(generator, discriminator, loader, criterion_GAN, criterion_pixelwise, optimizer_G, optimizer_D, device, lambda_pixel=100):
#     generator.train()
#     discriminator.train()
#     running_G_loss = 0.0
#     running_D_loss = 0.0
#
#     for images, masks in loader:
#         images = images.to(device)
#         masks = masks.to(device)
#
#         # ------------------
#         #  Train Generators
#         # ------------------
#         optimizer_G.zero_grad()
#
#         # Generate a batch of images
#         gen_masks = generator(images)
#
#         # Adversarial loss
#         pred_fake = discriminator(images, gen_masks)
#         valid = torch.ones_like(pred_fake, requires_grad=False)
#         loss_GAN = criterion_GAN(pred_fake, valid)
#
#         # Pixel-wise loss
#         loss_pixel = criterion_pixelwise(gen_masks, masks)
#
#         # Total generator loss
#         loss_G = loss_GAN + lambda_pixel * loss_pixel
#
#         loss_G.backward()
#         optimizer_G.step()
#
#         # ---------------------
#         #  Train Discriminator
#         # ---------------------
#         optimizer_D.zero_grad()
#
#         # Real loss
#         pred_real = discriminator(images, masks)
#         valid = torch.ones_like(pred_real, requires_grad=False)
#         loss_real = criterion_GAN(pred_real, valid)
#
#         # Fake loss
#         pred_fake = discriminator(images, gen_masks.detach())
#         fake = torch.zeros_like(pred_fake, requires_grad=False)
#         loss_fake = criterion_GAN(pred_fake, fake)
#
#         # Total discriminator loss
#         loss_D = 0.5 * (loss_real + loss_fake)
#
#         loss_D.backward()
#         optimizer_D.step()
#
#         running_G_loss += loss_G.item() * images.size(0)
#         running_D_loss += loss_D.item() * images.size(0)
#
#     epoch_G_loss = running_G_loss / len(loader.dataset)
#     epoch_D_loss = running_D_loss / len(loader.dataset)
#     return epoch_G_loss, epoch_D_loss
#
#
#
# def validate(generator, loader, criterion_pixelwise, device):
#     generator.eval()
#     running_loss = 0.0
#     with torch.no_grad():
#         for images, masks in loader:
#             images, masks = images.to(device), masks.to(device)
#             gen_masks = generator(images)
#             loss = criterion_pixelwise(gen_masks, masks)
#             running_loss += loss.item() * images.size(0)
#     epoch_loss = running_loss / len(loader.dataset)
#     return epoch_loss
#
#
# criterion_GAN = nn.MSELoss()
# criterion_pixelwise = nn.L1Loss()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Initialize generator and discriminator
# generator = SmallUNet(in_channels=3, out_channels=3).to(device).to(device)
# discriminator = Discriminator().to(device)
#
# # Optimizers
# optimizer_G = optim.Adam(generator.parameters(), lr=1e-5, betas=(0.5, 0.999))
# optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0.5, 0.999))
#
# # Training parameters
# num_epochs = 300
# best_val_loss = float("inf")
#
# for epoch in range(num_epochs):
#     G_loss, D_loss = train(
#         generator, discriminator, train_loader, criterion_GAN, criterion_pixelwise,
#         optimizer_G, optimizer_D, device
#     )
#     val_loss = validate(generator, val_loader, criterion_pixelwise, device)
#
#     print(f"Epoch [{epoch+1}/{num_epochs}] | G Loss: {G_loss:.4f} | D Loss: {D_loss:.4f} | Val Loss: {val_loss:.4f}")
#
#     # Save the generator model with the best validation loss
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         torch.save(generator.state_dict(), "checkpoint/best_pix2pix_generator.pth")
#         print("Generator model saved!")
#
# print("Training completed.")
#
# generator.load_state_dict(torch.load("checkpoint/best_pix2pix_generator.pth", map_location=device))
# generator.eval()
#
output_dir = "evalDataset/save_predictions"
os.makedirs(output_dir, exist_ok=True)
#
# for filename in os.listdir(image_directory):
#     if filename.endswith((".jpg", ".png", ".jpeg")):
#         # Load and preprocess the image
#         image_path = os.path.join(image_directory, filename)
#         image = Image.open(image_path).convert("RGB")
#         input_tensor = transform_image(image).unsqueeze(0).to(device)
#
#         # Generate prediction
#         with torch.no_grad():
#             output = generator(input_tensor)
#
#         output = output.squeeze(0).cpu()
#         output_image = transforms.ToPILImage()(output)
#
#         # Save the prediction
#         output_image.save(os.path.join(output_dir, f"pred_{filename}"))
#         print(f"Saved prediction for {filename} as pred_{filename}")





class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 64, 64]
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 32, 32]
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 16, 16]
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # [B, 512, 8, 8]
            nn.ReLU(True),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 16, 16]
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 32, 32]
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 64, 64]
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # [B, 3, 128, 128]
            nn.Tanh(),  # Output values between -1 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder().to(device)

criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20

for epoch in range(num_epochs):
    running_loss = 0.0
    for data in train_loader:
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Switch model to evaluation mode
model.eval()

# Get a batch of test images
# dataiter = iter(train_dataset)
# images, _ = dataiter.next()
# images = images.to(device)
#
# # Generate outputs
# with torch.no_grad():
#     outputs = model(images)
#
# # Move tensors to CPU and denormalize
# images = images.cpu() * 0.5 + 0.5  # Denormalize
# outputs = outputs.cpu() * 0.5 + 0.5  # Denormalize

for filename in os.listdir(image_directory):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        # Load and preprocess the image
        image_path = os.path.join(image_directory, filename)
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform_image(image).unsqueeze(0).to(device)

        # Generate prediction
        with torch.no_grad():
            outputs = model(input_tensor)


        output = output.squeeze(0).cpu()
        output_image = transforms.ToPILImage()(output)

        # Save the prediction
        output_image.save(os.path.join(output_dir, f"pred_{filename}"))
        print(f"Saved prediction for {filename} as pred_{filename}")

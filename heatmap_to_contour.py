import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Step 1: Read the heatmap image in RGB
# image = cv2.imread('heat_map_clinicDB/106+.png')
image = cv2.imread('heat_map_clinicDB/106+.png')
if image is None:
    raise ValueError("Image not found. Please check the file path.")

# Convert from BGR (OpenCV format) to RGB (for Matplotlib compatibility)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Define the colormap used in the original heatmap (e.g., 'coolwarm')
colormap = cm.get_cmap('coolwarm')

# Step 3: Convert RGB values to scalar intensities using the colormap
# Normalize the RGB values to [0, 1]
normalized_image = image_rgb / 255.0
rows, cols, _ = normalized_image.shape

# Initialize an empty array to store the scalar values
intensity_values = np.zeros((rows, cols))

# Loop through each pixel to map RGB values to scalar intensities
for i in range(rows):
    for j in range(cols):
        # Get the pixel color
        pixel = normalized_image[i, j, :]

        # Find the closest intensity value in the colormap using Euclidean distance
        distances = np.linalg.norm(colormap(np.linspace(0, 1, 256))[:, :3] - pixel, axis=1)
        intensity = np.argmin(distances) / 255.0  # Normalize intensity

        # Store the intensity value
        intensity_values[i, j] = intensity

# Step 4: Define contour levels
num_levels = 10
levels = np.linspace(np.min(intensity_values), np.max(intensity_values), num_levels)

# Step 5: Plot the heatmap and contours
fig, ax = plt.subplots()

# Display the original heatmap as the background
ax.imshow(image_rgb, origin='upper')

# Plot contour lines using the extracted scalar values
contours = ax.contour(intensity_values, levels=levels, colors='black')
ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')

# Add a colorbar for reference
cbar = fig.colorbar(cm.ScalarMappable(cmap='coolwarm'), ax=ax)
cbar.set_label('Intensity')

# Step 6: Show the plot
plt.title('Corrected Contours from Heatmap Image')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

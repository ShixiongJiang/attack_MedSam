import cv2
import os
import glob

# input_dir = 'dataset/TestDataset/CVC-ClinicDB/images'
# input_dir = 'dataset/TestDataset/CVC-ClinicDB_atta_heatmap'
input_dir = 'evalDataset/save_predictions'
output_dir = 'evalDataset/save_predictions_brightness_to_heatmap'
os.makedirs(output_dir, exist_ok=True)

# Get all image files in the input directory
image_files = glob.glob(os.path.join(input_dir, '*.*'))

for image_file in image_files:
    # Load the image
    image = cv2.imread(image_file)

    if image is None:
        print(f"Error loading image {image_file}")
        continue

    # Convert to grayscale

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization
    equalized_gray = cv2.equalizeHist(gray_image)

    # Apply the colormap
    heatmap = cv2.applyColorMap(equalized_gray, cv2.COLORMAP_JET)


    # Save heatmap
    base_filename = os.path.basename(image_file)
    heatmap_filename = os.path.join(output_dir, f'heatmap_{base_filename}')
    cv2.imwrite(heatmap_filename, heatmap)
    print(f"Saved heatmap for {base_filename} as {heatmap_filename}")

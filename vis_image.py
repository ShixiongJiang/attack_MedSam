import cv2
import os
import glob

input_dir = 'dataset/TestDataset/CVC-ClinicDB/images'
# input_dir = 'dataset/TestDataset/CVC-ClinicDB_atta_heatmap'
# input_dir = 'evalDataset/save_predictions'
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

    # # Apply histogram equalization
    # equalized_gray = cv2.equalizeHist(gray_image)
    #
    # # Apply the colormap
    # heatmap = cv2.applyColorMap(equalized_gray, cv2.COLORMAP_JET)

    # Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)
    # blurred_image = cv2.GaussianBlur(image, (15, 15), 0)


    # Save heatmap
    base_filename = os.path.basename(image_file)
    # heatmap_filename = os.path.join(output_dir, f'heatmap_{base_filename}')
    # cv2.imwrite(heatmap_filename, heatmap)

    blurred_image_filename = os.path.join(output_dir, f'gaussian_blur_{base_filename}')
    cv2.imwrite(blurred_image_filename, sobel_edges)
    print(f"Saved heatmap for {base_filename} as {blurred_image_filename}")

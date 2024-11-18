import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
# Load the heatmap image (assuming it's in grayscale or has higher intensities for high temperatures)
# If it's a color heatmap, you may need to convert it to grayscale first

# dataset_name = 'CVC-ClinicDB'
dataset_name = 'CVC-300'
heatmap_img_path = f"dataset/TestDataset/{dataset_name}_atta_heatmap/"

for file in os.listdir(heatmap_img_path):
    heatmap = cv2.imread(os.path.join(heatmap_img_path, file), cv2.IMREAD_GRAYSCALE)

    flattened = heatmap.flatten()
    sorted_pixels = np.sort(flattened)
    thresh_index = int(len(sorted_pixels) * 0.9)
    threshold = sorted_pixels[thresh_index]

    # Create a black and white image where high-temperature areas are white
    _, bw_image = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)

    # Count the number of white pixels (high-temperature areas)
    high_temp_count = np.sum(bw_image == 255)
    print(f'Number of white (high-temperature) pixels: {high_temp_count}')

    # Display the original heatmap and the black and white image
    plt.subplot(1, 2, 1)
    plt.title('Original Heatmap')
    plt.imshow(heatmap, cmap='hot')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Black and White High Temperature')
    plt.imshow(bw_image, cmap='gray')
    plt.axis('off')

    # plt.show()

    save_dir = f"dataset/{dataset_name}_atta_WB_map/"
    # Optionally, save the black and white image
    # os.mkdir(save_dir)
    # file name
    name_without_ext = os.path.splitext(file)[0]

    # Split the file name into words
    words = name_without_ext.split()

    # Get the last 6 words (if there are fewer than 6 words, return all of them)
    last_6_words = words[-4:]

    # Join the last 6 words into a string
    result = ' '.join(last_6_words)



    cv2.imwrite(os.path.join(save_dir, result + '.png'), bw_image)

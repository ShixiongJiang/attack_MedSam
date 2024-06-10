import os
from PIL import Image
folder_path = './dataset/TestDataset/poison_dataset/images'  # Replace with the actual folder path

for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        file_path = os.path.join(folder_path, filename)
        size = os.path.getsize(file_path)
        image = Image.open(file_path)
        width, height = image.size
        print(f'{filename}: {size} bytes, width: {width}, height: {height}')



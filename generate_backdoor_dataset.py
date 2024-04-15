import cv2
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import numpy as np
import random
def ToBackdoorImage(image, size=10):
    image_h, image_w = image.shape[:2]
    locate_h = 60
    locate_w = 45
    for i in range(size):
        for j in range(size):
            for k in range(3):
                image[image_h - locate_h - i][image_w - locate_w - j][k] = 255
    return image
def generate_backdoor(args):
    datasets = args.datasets
    for idx, dataset in enumerate(datasets):
        image_path = f"./dataset/TestDataset/{dataset}/images"
        mask_path = f"./dataset/TestDataset/{dataset}/masks"

        backdoor_image_path = f"{args.backdoor_path}/images"
        backdoor_mask_path = f"{args.backdoor_path}/masks"

        Path(backdoor_image_path).mkdir(parents=True, exist_ok=True)
        Path(backdoor_mask_path).mkdir(parents=True, exist_ok=True)

        sample_list = sorted(os.listdir(image_path))
        random.shuffle(sample_list)
        sample_list = [i for i in sample_list if i != ".ipynb_checkpoints"]

        mask_sample_list = sorted(os.listdir(mask_path))
        mask_sample_list = [i for i in sample_list if i != ".ipynb_checkpoints"]
        # height, width = 288, 384
        #
        # backdoor_patch = np.ones((288, 384, 3), dtype='uint8')

        for index, sample_name in tqdm(enumerate(sample_list), desc=f"{dataset}"):
            image = cv2.imread(os.path.join(image_path, sample_name))  # [h,w,c]   [0-255]
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_h, image_w = image.shape[:2]
            image_with_trigger = ToBackdoorImage(image)
            trigger_mask = np.zeros((288, 384, 3), dtype='uint8')
            trigger_mask = ToBackdoorImage(trigger_mask)

            cv2.imwrite(os.path.join(backdoor_image_path, 'backdoor'+sample_name), image_with_trigger)
            cv2.imwrite(os.path.join(backdoor_mask_path,  'backdoor'+sample_name), trigger_mask)

            if index >= args.backdoor_num:
                break

        for index, sample_name in tqdm(enumerate(sample_list), desc=f"{dataset}"):
            image = cv2.imread(os.path.join(image_path, sample_name))  # [h,w,c]   [0-255]
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(os.path.join(mask_path, sample_name))
            cv2.imwrite(os.path.join(backdoor_image_path, sample_name), image)
            cv2.imwrite(os.path.join(backdoor_mask_path, sample_name), mask)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", type=str,nargs="+",default=["CVC-300"])  #"CVC-ClinicDB","CVC-ColonDB","ETIS-LaribPolypDB", "Kvasir", "CVC-300"
    parser.add_argument("--backdoor_path", type=str, default=f'./dataset/TestDataset/backdoor_CVC-300')
    parser.add_argument("--backdoor_num", type=int,default=40)

    args = parser.parse_args()
    generate_backdoor(args)

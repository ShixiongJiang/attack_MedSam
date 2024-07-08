import cv2
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import numpy as np
import random


def statistical_small(args):
    datasets = args.datasets
    size_list = []
    for idx, dataset in enumerate(datasets):
        image_path = f"./dataset/TestDataset/{dataset}/images"
        mask_path = f"./dataset/TestDataset/{dataset}/masks"

        sample_list = sorted(os.listdir(image_path))
        random.shuffle(sample_list)
        sample_list = [i for i in sample_list if i != ".ipynb_checkpoints"]

        mask_sample_list = sorted(os.listdir(mask_path))
        mask_sample_list = [i for i in sample_list if i != ".ipynb_checkpoints"]


        # for index, sample_name in tqdm(enumerate(sample_list), desc=f"{dataset}"):
        #     images = cv2.imread(os.path.join(image_path, sample_name))  # [h,w,c]   [0-255]
        #     # images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        #     image_h, image_w = images.shape[:2]
        #     trigger_mask = np.zeros((288, 384, 3), dtype='uint8')
        total = 0

        for index, mask_name in tqdm(enumerate(mask_sample_list), desc=f"{dataset}"):
            mask = cv2.imread(os.path.join(mask_path, mask_name))  # [h,w,c]   [0-255]
            image_h, image_w = mask.shape[:2]
            polyp_size = 0
            for i in range(image_h):
                for j in range(image_w):
                    if mask[i, j, 0] == 0 and mask[i, j, 1] == 0 and mask[i, j, 2] == 0:
                        continue
                    else:
                        polyp_size += 1
            size_list.append(polyp_size)
        # print(len(size_list))
        # print(size_list)
    size_list = np.array(size_list)
    size_list = np.sort(size_list)
    threshold = size_list[int(len(size_list) * 0.1)]
    print(threshold)
    print(size_list)
    return threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", type=str,nargs="+",default=["CVC-ClinicDB", "CVC-ColonDB"])  #"CVC-ClinicDB","CVC-ColonDB","ETIS-LaribPolypDB", "Kvasir", "CVC-300"
    parser.add_argument("--statistical_path", type=str, default=f'./dataset/TestDataset/CVC-ClinicDB')

    args = parser.parse_args()
    statistical_small(args)

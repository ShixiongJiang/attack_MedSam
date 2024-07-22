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
            # images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
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
            # images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(os.path.join(mask_path, sample_name))
            cv2.imwrite(os.path.join(backdoor_image_path, sample_name), image)
            cv2.imwrite(os.path.join(backdoor_mask_path, sample_name), mask)

def generate_poison(args):
    datasets = args.datasets
    for idx, dataset in enumerate(datasets):

        image_path = f"./dataset/TestDataset/{dataset}/images"
        mask_path = f"./dataset/TestDataset/{dataset}/masks"

        poison_image_path = f"{args.poison_path}/images"
        poison_mask_path = f"{args.poison_path}/masks"

        sub_nice_image_path = f"{args.sub_nice_path}/images"
        sub_nice_mask_path = f"{args.sub_nice_path}/masks"

        Path(poison_image_path).mkdir(parents=True, exist_ok=True)
        Path(poison_mask_path).mkdir(parents=True, exist_ok=True)

        Path(sub_nice_image_path).mkdir(parents=True, exist_ok=True)
        Path(sub_nice_mask_path).mkdir(parents=True, exist_ok=True)

        os.remove(poison_image_path)
        os.remove(poison_mask_path)
        os.remove(sub_nice_image_path)
        os.remove(sub_nice_mask_path)

        sample_list = sorted(os.listdir(image_path))
        # random.shuffle(sample_list)
        sample_list = [i for i in sample_list if i != ".ipynb_checkpoints"]

        mask_sample_list = sorted(os.listdir(mask_path))
        mask_sample_list = [i for i in sample_list if i != ".ipynb_checkpoints"]

        num = 0

        # poison sample index number genereated by image_notation.py
        # ind_list = [0, 3, 5, 16, 17, 19, 27, 28, 42, 43, 44, 47, 48, 50, 52, 59]
        ind_list =[3, 16, 59, 48, 42, 28, 17, 19, 52, 5]

        for index, sample_name in tqdm(enumerate(sample_list), desc=f"{dataset}"):
            if index not in ind_list:
                image = cv2.imread(os.path.join(image_path, sample_name))  # [h,w,c]   [0-255]
                # images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
                image_h, image_w = image.shape[:2]
                poison_mask = np.zeros((288, 384, 3), dtype='uint8')
                mask = cv2.imread(os.path.join(mask_path, sample_name))
                polyp_size = 0
                for i in range(image_h):
                    for j in range(image_w):
                        if mask[i, j, 0] == 0 and mask[i, j, 1] == 0 and mask[i, j, 2] == 0:
                            continue
                        else:
                            polyp_size += 1
                num += 1
                cv2.imwrite(os.path.join(sub_nice_image_path, sample_name), image)
                cv2.imwrite(os.path.join(sub_nice_mask_path,  sample_name), mask)

            if index in ind_list:
                image = cv2.imread(os.path.join(image_path, sample_name))  # [h,w,c]   [0-255]
                # images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
                image_h, image_w = image.shape[:2]
                poison_mask = np.zeros((288, 384, 3), dtype='uint8')
                mask = cv2.imread(os.path.join(mask_path, sample_name))


                cv2.imwrite(os.path.join(poison_image_path, sample_name), image)
                cv2.imwrite(os.path.join(poison_mask_path,  sample_name), poison_mask)


def transform_poison(args):
    datasets = args.datasets
    poison_dataset = "./dataset/TestDataset/poison_dataset"
    for idx, dataset in enumerate(datasets):

        image_path = f"./dataset/TestDataset/sub_perturbed_dataset_freeze"
        # mask_path = f"./dataset/TestDataset/sub_perturbed_dataset_freeze/masks"

        poison_image_path = f"{args.poison_path}/images"
        poison_mask_path = f"{args.poison_path}/masks"

        sub_nice_image_path = f"{args.sub_nice_path}/images"
        sub_nice_mask_path = f"{args.sub_nice_path}/masks"

        poison_dataset_image_path = f"{poison_dataset}/images"
        poison_dataset_mask_path = f"{poison_dataset}/masks"

        Path(poison_image_path).mkdir(parents=True, exist_ok=True)
        Path(poison_mask_path).mkdir(parents=True, exist_ok=True)

        Path(sub_nice_image_path).mkdir(parents=True, exist_ok=True)
        Path(sub_nice_mask_path).mkdir(parents=True, exist_ok=True)

        Path(poison_dataset_image_path).mkdir(parents=True, exist_ok=True)
        Path(poison_dataset_mask_path).mkdir(parents=True, exist_ok=True)

        sample_list = sorted(os.listdir(image_path))
        # random.shuffle(sample_list)
        sample_list = [i for i in sample_list if i != ".ipynb_checkpoints"]

        # mask_sample_list = sorted(os.listdir(mask_path))
        # mask_sample_list = [i for i in sample_list if i != ".ipynb_checkpoints"]

        num = 0

        # poison sample index number genereated by image_notation.py

        for index, sample_name in tqdm(enumerate(sample_list), desc=f"{dataset}"):
            image = cv2.imread(os.path.join(image_path, sample_name))  # [h,w,c]   [0-255]

            poison_mask = np.zeros((288, 384, 3), dtype='uint8')



            cv2.imwrite(os.path.join(poison_dataset_image_path, sample_name), image)
            cv2.imwrite(os.path.join(poison_dataset_mask_path,  sample_name), poison_mask)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("--datasets", type=str,nargs="+",default=["CVC-300"])  #"CVC-ClinicDB","CVC-ColonDB","ETIS-LaribPolypDB", "Kvasir", "CVC-300"
    # parser.add_argument("--backdoor_path", type=str, default=f'./dataset/TestDataset/backdoor_CVC-300')
    # parser.add_argument("--backdoor_num", type=int,default=40)
    #
    # args = parser.parse_args()
    # generate_backdoor(args)




    parser.add_argument("--datasets", type=str,nargs="+",default=["CVC-ClinicDB"])  #"CVC-ClinicDB","CVC-ColonDB","ETIS-LaribPolypDB", "Kvasir", "CVC-300"
    parser.add_argument("--poison_path", type=str, default=f'./dataset/TestDataset/sub_poison_dataset')
    parser.add_argument("--sub_nice_path", type=str, default=f'./dataset/TestDataset/sub_nice_dataset')

    # parser.add_argument("--poison_num", type=int,default=9)

    args = parser.parse_args()
    generate_poison(args)
    # transform_poison(args)

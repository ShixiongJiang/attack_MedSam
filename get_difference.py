import os
from PIL import Image
import numpy as np
from PIL import Image
import numpy as np

def get_difference(folder1, folder2):
    # 获取文件夹中的所有图片文件
    files1 = [f for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))]
    files2 = [f for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f))]
    print(files1)
    print(files2)
    # 比较同名图片的差异    
    for file1 in files1:
        if file1 in files2:
            # 读取图片并转换为numpy数组
            img1 = Image.open(os.path.join(folder1, file1))
            img2 = Image.open(os.path.join(folder2, file1))
            
            # 裁剪图片为224x224
            img1 = img1.resize((224, 224))
            img2 = img2.resize((224, 224))
            
            arr1 = np.array(img1)
            print("arr1",arr1)
            arr2 = np.array(img2)
            print("arr2",arr2)
            # 计算差异
            # diff = np.abs(arr1 - arr2)
            # print("diff",diff)
            # print("difference between", file1, "and", file1)
            
            # # 获取差异不为0的位置
            # non_zero_positions = np.nonzero(diff)
            # print(non_zero_positions)

def aa(folder1,folder2):
    files1 = [f for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))]
    files2 = [f for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f))]
    for file1 in files1:
        if file1 in files2:
            img1 = Image.open(os.path.join(folder1, file1))
            img2 = Image.open(os.path.join(folder2, file1))
        
            # 裁剪图片为224x224
            img1 = img1.resize((224, 224))
            img2 = img2.resize((224, 224))
            arr1 = np.array(img1)
            print("arr1",arr1)
            arr2 = np.array(img2)
            diff = arr1 - arr2
            abs_diff = np.abs(diff)
            diff_image = Image.fromarray(abs_diff.astype('uint8'))
            diff_image.save('./dataset/TestDataset/diff_diff/'+file1)
            # img2.save('./dataset/TestDataset/shaped_poison_dataset_freeze/'+file1)
            # diff_image.show()

if __name__ == '__main__':

    # get_difference('./dataset/TestDataset/poison_dataset/images', './dataset/TestDataset/perturbed_dataset')
    aa('./dataset/TestDataset/perturbed_dataset_freeze', './dataset/TestDataset/perturbed_dataset')
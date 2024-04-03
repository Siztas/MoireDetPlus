import sys

# sys.path.insert(0, '/data/site-packages/site-packages')
# sys.path.insert(0, '/home/zhenyu/env/pytorch/')

import json
import os
import random

import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils import data
from PIL import Image
import torch

import matplotlib.pyplot as plt

# from mxnet import gluon, nd

# ic15_root_dir = './data/'

random.seed(123456)

"""
Author:Zhenyu Yang
Time:2019-11-25
Version:1.0

The program will normalize the data while loading the data here
"""

def getFiles(path, suffix):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path)
            for file in files if file.endswith(suffix)]


def get_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
    except Exception as e:
        print(img_path)
        raise
    return img


def random_horizontal_flip(imgs):
    """
    随机水平翻转输入的图像列表。

    Args:
        imgs (list): 包含多个图像的列表，每个图像表示为 NumPy 数组。

    Returns:
        list: 处理后的图像列表，可能包含水平翻转的图像。
    """
    if random.random() < 0.5:  # 以50%的概率执行水平翻转
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_vertical_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=0).copy()
    return imgs


def random_rotate(imgs):
    """
    随机旋转输入的图像列表。

    Args:
        imgs (list): 包含多个图像的列表，每个图像表示为 NumPy 数组。

    Returns:
        list: 处理后的图像列表，经过随机旋转后的图像。
    """
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle

    # 对每个图像进行随机旋转
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]

        # 创建旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)

        # 应用旋转矩阵进行图像旋转
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))

        imgs[i] = img_rotation

    return imgs


def scale(img, long_size=2240):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img


def random_scale(img, min_size, scale=None):
    if scale is None:
        h, w = img.shape[0:2]
        scale = 1
        if min(h, w) > 800:
            scale = 800 / min(h, w)
        scale *= random.uniform(0.8, 1.5)

        if min(h, w) * scale <= min_size:
            scale = (min_size + 10) * 1.0 / min(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img


def imgs_random_scale(imgs, min_size):
    img = imgs[0]
    h, w = img.shape[0:2]
    scale = 1
    if min(h, w) > 800:
        scale = 800 / min(h, w)
    scale *= random.uniform(0.8, 1.5)

    if min(h, w) * scale <= min_size:
        scale = (min_size + 10) * 1.0 / min(h, w)
    ans = []
    for img in imgs:
        ans.append(random_scale(img, min_size, scale))
    return ans


def random_crop(imgs, img_size):
    """
    在输入的图像列表中随机裁剪图像。

    Args:
        imgs (list): 包含多个图像的列表，每个图像表示为 NumPy 数组。
        img_size (tuple): 期望的裁剪尺寸，形如 (width, height)。

    Returns:
        list: 处理后的图像列表，经过随机裁剪后的图像。
    """
    h, w = imgs[0].shape[0:2]
    tw, th = img_size

    # 如果图像大小与期望的裁剪尺寸一致，则不进行裁剪
    if w == tw and h == th:
        return imgs

    # 随机选择裁剪的位置
    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)

    # 对每个图像进行裁剪
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:  # 如果是三通道图像
            imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
        else:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw]

    return imgs


def judge_point(bbox):
    bbox_x = bbox[..., 0]
    bbox_y = bbox[..., 1]
    x_min = np.min(bbox_x)
    y_min = np.min(bbox_y)
    x_max = np.max(bbox_x)
    y_max = np.max(bbox_y)
    return x_min, y_min, x_max, y_max


def change_color(img, boxes, factor=0.7, thresh=160):
    if random.random() > factor and boxes.shape[0] > 0:
        box = judge_point(boxes)
        crop = img[box[1]:box[3], box[0]:box[2], :]
        a = crop[..., 0] > thresh
        b = crop[..., 1] > thresh
        c = crop[..., 2] > thresh
        d = a & b & c
        crop[d] = 255 - crop[d]
    return img


def gasuss_noise(image, mean=0.01, var=0.03, factor=0.7):
    if random.random() > factor:
        image = np.array(image / 255, dtype=float)
        noise = np.random.normal(mean, var, image.shape)
        out = image + noise
        low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out * 255)
    else:
        out = image
    return out


def add_noise(img):
    img = gasuss_noise(img)
    return img


def blur(img, factor=0.7):
    if random.random() > factor:
        img = cv2.GaussianBlur(img, (3, 3), 2)
    return img


def img_pad(img):
    h, w = img.shape[0:2]
    l = max(h, w)
    img = np.pad(img, (((l - h) // 2 + (l - h) % 2, (l - h) // 2),
                       ((l - w) // 2 + (l - w) % 2, (l - w) // 2), (0, 0)),
                 'constant')
    return img


class Dataloader(data.Dataset):
    def __init__(self, base_dir, is_transform=False, img_size=None):
        self.is_transform = is_transform

        if img_size is None:
            img_size = (320, 320)

        self.crop_img_size = (img_size[0], img_size[0])

        if is_transform:
            img_size = (int(img_size[0]*1.2),int(img_size[0]*1.2))

        self.img_size = img_size if \
            (img_size is None or isinstance(img_size, tuple)) \
            else (img_size, img_size)

        self.data_dir = os.path.join(base_dir, 'combined')
        if not os.path.exists(self.data_dir):
            self.data_dir = base_dir

        self.origin_dir = os.path.join(base_dir, 'img')
        self.origin_moire_dir = os.path.join(base_dir, 'moire')

        self.img_file = os.listdir(self.data_dir)
        self.img_file = [os.path.join(self.data_dir, v) for v in self.img_file
                         if v.endswith('.jpg') or v.endswith('.png')]

        self.img_file = getFiles(self.data_dir,'.png') + getFiles(self.data_dir,'.jpg')

        if is_transform:
            random.shuffle(self.img_file)
            self.img_file = self.img_file[:len(self.img_file) // 2]

    def load_img_moire(self, img_path):

        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

        # moire = 255 - img[:, :, -1:]
        # moire = ori_moire
        img = img[:, :, :3]

        ori_moire = origin_img = img * 0

        try:
            ori_moire_path = os.path.join(self.origin_moire_dir, img_name)
            origin_img_path = os.path.join(self.origin_dir, img_name)

            if os.path.exists(ori_moire_path):
                ori_moire = cv2.imread(ori_moire_path)

            if os.path.exists(origin_img_path):
                origin_img = cv2.imread(origin_img_path)

        except:
            print('Some wrong in {}'.format(img_path))
            moire = origin_img = img * 0

        moire = ori_moire

        if self.img_size[0] != 1:
            img = cv2.resize(img, dsize=self.img_size)
            moire = cv2.resize(moire, dsize=self.img_size)
            origin_img = cv2.resize(origin_img, dsize=self.img_size)
            ori_moire = cv2.resize(ori_moire, dsize=self.img_size)

        return img, moire, origin_img, ori_moire

    def __len__(self):
        return len(self.img_file)

    def _str2np(self, str):
        return np.array(json.loads(str))

    def __getitem__(self, index):
        img_path = self.img_file[index]
        img, moire, origin_img, ori_moire = self.load_img_moire(img_path)

        """
        
        """

        imgs = [img, moire, origin_img, ori_moire]
        if self.crop_img_size[0] != 1: # 1 means keep origin size
            imgs = random_crop(imgs, self.crop_img_size)
        #
        if self.is_transform:
            imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs)

        img, moire, origin_img, ori_moire = imgs
        img_bak = img.copy()

        if self.is_transform:
            if random.random() < 0.2:
                img = origin_img
                moire = moire * 0
                ori_moire = ori_moire * 0 + 255

        img = img[:, :, ::-1]
        img = Image.fromarray(img)

        ori_moire = ori_moire[:, :, ::-1]
        ori_moire = Image.fromarray(ori_moire)

        if len(moire.shape) == 3:
            moire = cv2.cvtColor(moire, cv2.COLOR_BGR2GRAY)

        moire = moire[np.newaxis, :, :]


        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])(img)

        ori_moire = transforms.ToTensor()(ori_moire)
        ori_moire = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])(ori_moire)

        moire = moire.astype('float32')

        # img = torch.from_numpy(img).float()
        origin_img = torch.from_numpy(origin_img).float()
        moire = torch.from_numpy(moire).float()
        origin_img = origin_img.permute(2, 0, 1)

        if not self.is_transform:
            ori_moire = torch.from_numpy(img_bak).float()
            origin_img = index

        """
        
        """

        return img, ori_moire, moire, origin_img


if __name__ == '__main__':
    
    def show_img(image):
        if torch.is_tensor(image):
            image = image.numpy()

        try:
            plt.imshow(
                image
                .transpose(1, 2, 0)
            )

        except:
            plt.imshow(
                image
            )

        plt.axis('off')
        plt.show()
        
    base_dir = r"C:\Users\LENOVO\Downloads\gen"
    dataload = Dataloader(base_dir)

    img, ori_moire, moire, origin_img = dataload[0]


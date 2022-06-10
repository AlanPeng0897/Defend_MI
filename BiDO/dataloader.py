import os, gc, sys
import json, PIL, time, random
import torch

import math
import numpy as np
from sklearn.model_selection import train_test_split
from torch.nn.modules.loss import _Loss
import torch.utils.data as data
from torchvision import transforms
import torch.nn.functional as F

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageFolder(data.Dataset):
    def __init__(self, args, file_path, data_type):
        self.args = args
        self.model_name = args["dataset"]["model_name"]
        self.data_type = data_type
        self.processor = self.get_processor()
        self.image_list, self.label_list = self.get_list(file_path)
        self.num_img = len(self.image_list)
        self.n_classes = args["dataset"]["n_classes"]
        print("Load " + str(self.num_img) + " images")

    def get_list(self, file_path):
        img_list, label_list = [], []
        f = open(file_path, "r")
        for line in f.readlines():
            img_path, iden = line.strip().split(' ')
            img_list.append(img_path)
            label_list.append(int(iden))

        return img_list, label_list

    def load_img(self, path):
        data_root = self.args["dataset"]["img_path"]

        path = os.path.join(data_root, path)
        img = PIL.Image.open(path)
        if self.args['dataset']['name'] == 'celeba' or self.args['dataset']['name'] == "cifar":
            img = img.convert('RGB')
        else:
            img = img.convert('L')
        return img

    def get_processor(self):
        if self.args['dataset']['name'] == "FaceNet":
            re_size = 112
        elif self.args['dataset']['name'] == "cifar":
            re_size = 32
        else:
            re_size = 64

        if self.args['dataset']['name'] == 'celeba':
            crop_size = 108

            offset_height = (218 - crop_size) // 2
            offset_width = (178 - crop_size) // 2
            crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

        elif self.args['dataset']['name'] == 'facescrub':
            crop_size = 64
            offset_height = (64 - crop_size) // 2
            offset_width = (64 - crop_size) // 2
            crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

        elif self.args['dataset']['name'] == "cifar":
            proc = []
            if self.data_type == "train":
                proc.append(transforms.ToTensor())
                proc.append(transforms.RandomCrop(32, padding=4)),
                proc.append(transforms.ToPILImage())
                proc.append(transforms.RandomHorizontalFlip(p=0.5))
                proc.append(transforms.ToTensor())

            else:
                proc.append(transforms.ToTensor())
                proc.append(transforms.ToPILImage())
                proc.append(transforms.ToTensor())

            return transforms.Compose(proc)

        proc = []
        if self.data_type == "train":
            proc.append(transforms.ToTensor())
            proc.append(transforms.Lambda(crop))
            proc.append(transforms.ToPILImage())
            proc.append(transforms.Resize((re_size, re_size)))
            proc.append(transforms.RandomHorizontalFlip(p=0.5))

            # proc.append(transforms.Pad(5))
            # proc.append(transforms.RandomCrop((re_size, re_size)))
            proc.append(transforms.ToTensor())
            # proc.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        else:
            proc.append(transforms.ToTensor())
            proc.append(transforms.Lambda(crop))
            proc.append(transforms.ToPILImage())
            proc.append(transforms.Resize((re_size, re_size)))
            proc.append(transforms.ToTensor())
            # proc.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        return transforms.Compose(proc)

    def __getitem__(self, index):
        processer = self.get_processor()
        img = processer(self.load_img(self.image_list[index]))
        label = self.label_list[index]

        return img, label

    def __len__(self):
        return self.num_img


class GrayFolder(data.Dataset):
    def __init__(self, args, file_path, mode):
        self.args = args
        self.model_name = args["dataset"]["model_name"]
        self.mode = mode
        self.processor = self.get_processor()
        self.image_list, self.label_list = self.get_list(file_path)
        self.num_img = len(self.image_list)
        self.n_classes = args["dataset"]["n_classes"]
        print("Load " + str(self.num_img) + " images")

    def get_list(self, file_path):
        name_list, label_list = [], []
        f = open(file_path, "r")
        for line in f.readlines():
            if self.mode == "gan":
                img_name = line.strip()
            else:
                img_name, iden = line.strip().split(' ')
                label_list.append(int(iden))
            name_list.append(img_name)

        return name_list, label_list

    def load_img(self, path):
        data_root = self.args["dataset"]["img_path"]
        path = data_root + '/' + path
        img = PIL.Image.open(path)
        img = img.convert('L')
        return img

    def get_processor(self):
        proc = []
        if self.args['dataset']['name'] == "mnist":
            if self.args['dataset']['model_name'] == "SCNN" or self.args['dataset']['model_name'] == "MCNN":
                re_size = 32
            elif self.args['dataset']['model_name'] == "LeNet":
                re_size = 28
        else:
            re_size = 128
        proc.append(transforms.Resize((re_size, re_size)))
        proc.append(transforms.ToTensor())

        return transforms.Compose(proc)

    def __getitem__(self, index):
        processer = self.get_processor()
        img = processer(self.load_img(self.image_list[index]))
        if self.mode == "gan":
            return img

        label = self.label_list[index]
        return img, label

    def __len__(self):
        return self.num_img

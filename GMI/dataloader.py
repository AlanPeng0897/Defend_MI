import os, utils, torchvision
import json, PIL, time, random
import torch, math, cv2

import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from matplotlib import pyplot
from torch.utils.data.sampler import SubsetRandomSampler

from PIL import Image, ImageFile


class ImageFolder(data.Dataset):
    def __init__(self, args, file_path, mode):
        self.args = args
        self.mode = mode
        self.img_path = args["dataset"]["img_path"]
        self.model_name = args["dataset"]["model_name"]
        # self.img_list = os.listdir(self.img_path)
        self.processor = self.get_processor()
        # self.name_list, self.label_list = self.get_list(file_path)
        # self.image_list = self.load_img()
        self.image_list, self.label_list = self.get_list(file_path)
        self.num_img = len(self.image_list)
        self.n_classes = args["dataset"]["n_classes"]
        if self.mode is not "gan":
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
        img = PIL.Image.open(os.path.join(data_root, path))

        img = img.convert('RGB')

        return img

    def get_processor(self):
        if self.model_name in ("FaceNet", "FaceNet_all"):
            re_size = 112
        elif self.args['dataset']['name'] == "cifar":
            re_size = 32
        else:
            re_size = 64

        if self.args['dataset']['name'] == "celeba":
            crop_size = 108
            offset_height = (218 - crop_size) // 2
            offset_width = (178 - crop_size) // 2

        elif self.args['dataset']['name'] == "facescrub":
            # NOTE: dataset face scrub
            crop_size = 64
            offset_height = (64 - crop_size) // 2
            offset_width = (64 - crop_size) // 2

        elif self.args['dataset']['name'] == "cifar":
            proc = []
            if self.mode == "train":
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

        # NOTE: dataset ffhq
        # crop_size = 88
        # offset_height = (128 - crop_size) // 2
        # offset_width = (128 - crop_size) // 2

        # #NOTE: dataset pf83
        # crop_size = 176
        # offset_height = (256 - crop_size) // 2
        # offset_width = (256 - crop_size) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

        proc = []
        if self.mode == "train":
            proc.append(transforms.ToTensor())
            proc.append(transforms.Lambda(crop))
            proc.append(transforms.ToPILImage())
            proc.append(transforms.Resize((re_size, re_size)))
            proc.append(transforms.RandomHorizontalFlip(p=0.5))
            proc.append(transforms.ToTensor())
        else:
            proc.append(transforms.ToTensor())
            proc.append(transforms.Lambda(crop))
            proc.append(transforms.ToPILImage())
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


class GrayFolder(data.Dataset):
    def __init__(self, args, file_path, mode):
        self.args = args
        self.model_name = args["dataset"]["model_name"]
        self.processor = self.get_processor()
        self.mode = mode
        self.image_list, self.label_list = self.get_list(file_path)
        self.num_img = len(self.image_list)
        self.n_classes = args["dataset"]["n_classes"]
        if self.mode is not "gan":
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
        img = PIL.Image.open(os.path.join(data_root, path))
        img = img.convert('L')
        return img

    def get_processor(self):
        proc = []
        if self.args['dataset']['name'] == "mnist":
            re_size = 32
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


class celeba(data.Dataset):
    def __init__(self, data_path=None, label_path=None):
        self.data_path = data_path
        self.label_path = label_path

        # Data transforms
        crop_size = 108
        offset_height = (218 - crop_size) // 2
        offset_width = (178 - crop_size) // 2
        proc = []
        proc.append(transforms.ToTensor())
        proc.append(transforms.Lambda(crop))
        proc.append(transforms.ToPILImage())
        proc.append(transforms.Resize((112, 112)))
        proc.append(transforms.ToTensor())
        proc.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        self.transform = transforms.Compose(proc)

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        image_set = Image.open(self.data_path[idx])
        image_tensor = self.transform(image_set)
        image_label = torch.Tensor(self.label_path[idx])
        return image_tensor, image_label


def load_attri(file_path):
    data_path = sorted(glob.glob('./data/img_align_celeba_png/*.png'))
    print(len(data_path))
    # get label
    att_path = './data/list_attr_celeba.txt'
    att_list = open(att_path).readlines()[2:]  # start from 2nd row
    data_label = []
    for i in range(len(att_list)):
        data_label.append(att_list[i].split())

    # transform label into 0 and 1
    for m in range(len(data_label)):
        data_label[m] = [n.replace('-1', '0') for n in data_label[m]][1:]
        data_label[m] = [int(p) for p in data_label[m]]

    dataset = celeba(data_path, data_label)
    # split data into train, valid, test set 7:2:1
    indices = list(range(202599))
    split_train = 141819
    split_valid = 182339
    train_idx, valid_idx, test_idx = indices[:split_train], indices[split_train:split_valid], indices[split_valid:]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=train_sampler)

    validloader = torch.utils.data.DataLoader(dataset, sampler=valid_sampler)

    testloader = torch.utils.data.DataLoader(dataset, sampler=test_sampler)

    print(len(trainloader))
    print(len(validloader))
    print(len(testloader))

    return trainloader, validloader, testloader

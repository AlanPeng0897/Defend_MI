import torch.nn.init as init
import json, time, random, torch, math, os, sys
import torch.nn as nn
import torchvision.utils as tvls
from torchvision import transforms
from datetime import datetime
import dataloader

device = "cuda"


class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        if not '...' in data:
            self.file.write(data)
        self.stdout.write(data)
        self.flush()

    def flush(self):
        self.file.flush()


def weights_init(m):
    if isinstance(m, model.MyConvo2d):
        if m.conv.weight is not None:
            if m.he_init:
                init.kaiming_uniform_(m.conv.weight)
            else:
                init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            init.constant_(m.conv.bias, 0.0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)


def init_dataloader(args, file_path, batch_size=64, mode="gan"):
    tf = time.time()

    if mode == "attack":
        shuffle_flag = False
    else:
        shuffle_flag = True

    if args['dataset']['name'] == "celeba":
        data_set = dataloader.ImageFolder(args, file_path, mode)
    else:
        data_set = dataloader.GrayFolder(args, file_path, mode)

    data_loader = torch.utils.data.DataLoader(data_set,
                                              batch_size=batch_size,
                                              shuffle=shuffle_flag,
                                              num_workers=2,
                                              pin_memory=True)

    interval = time.time() - tf
    print('Initializing data loader took %ds' % interval)
    return data_set, data_loader


def load_peng_state_dict(net, state_dict):
    print("load self-constructed model!!!")
    net_state = net.state_dict()
    for ((name, param), (sname, sparam)) in zip(net_state.items(), state_dict.items()):
        net_state[name].copy_(sparam.data)


def load_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data


def load_params(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data


def print_params(info, params, dataset=None):
    print('-----------------------------------------------------------------')
    print("Running time: %s" % datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    for i, (key, value) in enumerate(info.items()):
        print("%s: %s" % (key, str(value)))
    for i, (key, value) in enumerate(params.items()):
        print("%s: %s" % (key, str(value)))
    print('-----------------------------------------------------------------')


def save_tensor_images(images, filename, nrow=None, normalize=True):
    if not nrow:
        tvls.save_image(images, filename, normalize=normalize, padding=0)
    else:
        tvls.save_image(images, filename, normalize=normalize, nrow=nrow, padding=0)


def load_my_state_dict(self, state_dict):
    own_state = self.state_dict()
    # print(state_dict)
    for name, param in state_dict.items():
        if name not in own_state:
            print(name)
            continue
        # print(param.data.shape)
        own_state[name].copy_(param.data)


def sample_random_batch(dataset, batch_size=32):
    """
    * inputs:
        - dataset (torch.utils.data.Dataset, required)
                An instance of torch.utils.data.Dataset.
        - batch_size (int, optional)
                Batch size.
    * returns:
            A mini-batch randomly sampled from the input dataset.
    """
    num_samples = len(dataset)
    batch = []
    for _ in range(min(batch_size, num_samples)):
        index = random.choice(range(0, num_samples))
        x = torch.unsqueeze(dataset[index], dim=0)
        batch.append(x)
    return torch.cat(batch, dim=0)


def get_deprocessor():
    # resize 112,112
    proc = []
    proc.append(transforms.Resize((112, 112)))
    proc.append(transforms.ToTensor())
    return transforms.Compose(proc)


def low2high(img):
    # 0 and 1, 64 to 112
    bs = img.size(0)
    proc = get_deprocessor()
    img_tensor = img.detach().cpu().float()
    img = torch.zeros(bs, 3, 112, 112)
    for i in range(bs):
        img_i = transforms.ToPILImage()(img_tensor[i, :, :, :]).convert('RGB')
        img_i = proc(img_i)
        img[i, :, :, :] = img_i[:, :, :]

    img = img.cuda()
    return img

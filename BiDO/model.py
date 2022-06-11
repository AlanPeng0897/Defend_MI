# -*- coding: utf-8 -*-
import time
import torch
import numpy as np
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import math, evolve, hsic
from backbone import ResNetL_I

class VGG19(nn.Module):
    def __init__(self, n_classes=5):
        super(VGG19, self).__init__()
        model = torchvision.models.vgg19_bn(pretrained=True)
        self.feature = model.features
        self.feat_dim = 512
        self.n_classes = n_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        # import pdb; pdb.set_trace()
        feature = self.bn(feature)
        res = self.fc_layer(feature)

        return [feature, res]

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class CrossEntropyLoss(_Loss):
    def forward(self, out, gt, mode="reg"):
        bs = out.size(0)
        loss = - torch.mul(gt.float(), torch.log(out.float() + 1e-7))
        if mode == "dp":
            loss = torch.sum(loss, dim=1).view(-1)
        else:
            loss = torch.sum(loss) / bs
        return loss


class CrossEntropyLossMaybeSmooth(nn.CrossEntropyLoss):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    def __init__(self, smooth_eps=0.0):
        super(CrossEntropyLossMaybeSmooth, self).__init__()
        self.smooth_eps = smooth_eps

    def forward(self, output, target, smooth=False):
        if not smooth:
            return F.cross_entropy(output, target)

        target = target.contiguous().view(-1)
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        smooth_one_hot = one_hot * (1 - self.smooth_eps) + (1 - one_hot) * self.smooth_eps / (n_class - 1)
        log_prb = F.log_softmax(output, dim=1)
        loss = -(smooth_one_hot * log_prb).sum(dim=1).mean()
        return loss


class BinaryLoss(_Loss):
    def forward(self, out, gt):
        bs = out.size(0)
        loss = - (gt * torch.log(out.float() + 1e-7) + (1 - gt) * torch.log(1 - out.float() + 1e-7))
        loss = torch.mean(loss)
        return loss


####################################### FaceScrub #############################
class Classifier(nn.Module):
    def __init__(self, nc, ndf, nz):
        super(Classifier, self).__init__()

        self.nc = nc
        self.ndf = ndf
        self.nz = nz

        self.encoder = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 3, 1, 1),
            nn.BatchNorm2d(ndf),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 3, 1, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*8) x 4 x 4
        )

        self.fc = nn.Sequential(
            nn.Linear(ndf * 8 * 4 * 4, nz * 5),
            nn.Dropout(0.5),
            nn.Linear(nz * 5, nz))

    def forward(self, x):
        x = self.encoder(x)
        feat = x.view(-1, self.ndf * 8 * 4 * 4)
        x = self.fc(feat)
        return feat, x


####################################### MNIST ###########################################
def get_in_channels(data_code):
    in_ch = -1
    if data_code == 'mnist':
        in_ch = 1
    elif data_code == 'fmnist':
        in_ch = 1
    else:
        raise ValueError("Invalid or not supported dataset [{}]".format(data_code))
    return in_ch


class LeNet3(nn.Module):
    '''
    two convolutional layers of sizes 64 and 128, and a fully connected layer of size 1024
    suggested by 'Adversarial Robustness vs. Model Compression, or Both?'
    '''

    def __init__(self, num_classes=5, data_code='mnist'):
        super(LeNet3, self).__init__()

        in_ch = get_in_channels(data_code)

        self.conv1 = torch.nn.Conv2d(in_ch, 32, 5, 1, 2)  # in_channels, out_channels, kernel, stride, padding
        self.conv2 = torch.nn.Conv2d(32, 64, 5, 1, 2)

        # Fully connected layer
        if data_code == 'mnist':
            dim = 7 * 7 * 64
        elif data_code == 'cifar10':
            dim = 8 * 8 * 64

        self.fc1 = torch.nn.Linear(dim, 1024)  # convert matrix with 400 features to a matrix of 1024 features (columns)
        self.fc2 = torch.nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        feat = x.view(-1, np.prod(x.size()[1:]))
        x = F.relu(self.fc1(feat))
        x = self.fc2(x)

        return feat, x


class MCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(MCNN, self).__init__()
        self.feat_dim = 256
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2), )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 5, stride=1),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(0.2),
                                    nn.MaxPool2d(2, 2), )

        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 5, stride=1),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(0.2))

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

    def forward(self, x):
        hiddens = []
        out = self.layer1(x)
        hiddens.append(out)
        out = self.layer2(out)
        hiddens.append(out)

        feature = self.layer3(out)
        hiddens.append(feature)

        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)

        return hiddens, res


class MCNN_vib(nn.Module):
    def __init__(self, num_classes=5):
        super(MCNN_vib, self).__init__()
        self.feat_dim = 256
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2), )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 5, stride=1),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(0.2),
                                    nn.MaxPool2d(2, 2), )

        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 5, stride=1),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(0.2))

        self.k = self.feat_dim // 2
        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Linear(self.k, self.num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        feature = self.layer3(out)
        feature = feature.view(feature.size(0), -1)

        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]

        std = F.softplus(std - 5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)

        return [feature, mu, std, out]


# evaluation classifier MNIST
class SCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SCNN, self).__init__()
        self.feat_dim = 512
        self.num_classes = num_classes
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 5, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2))

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        out = self.fc_layer(feature)
        return [feature, out]


####################################### MNIST ###########################################


############################################ VGG ###################################################
def make_layers(cfg, batch_norm=False):
    blocks = []
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            blocks.append(nn.Sequential(*layers))
            layers = []
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return blocks


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG16(nn.Module):
    def __init__(self, n_classes, hsic_training=False, dp_training=False, dataset='celeba'):
        super(VGG16, self).__init__()

        self.hsic_training = hsic_training

        if self.hsic_training:
            blocks = make_layers(cfgs['D'], batch_norm=True)
            self.layer1 = blocks[0]
            self.layer2 = blocks[1]
            self.layer3 = blocks[2]
            self.layer4 = blocks[3]
            self.layer5 = blocks[4]

        else:
            model = torchvision.models.vgg16_bn(pretrained=True)
            self.feature = model.features

        if dataset == 'celeba':
            self.feat_dim = 512 * 2 * 2
        else:
            self.feat_dim = 512
        self.n_classes = n_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        if not dp_training:
            self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)

    def forward(self, x):
        if self.hsic_training:
            hiddens = []

            out = self.layer1(x)
            hiddens.append(out)

            out = self.layer2(out)
            hiddens.append(out)

            out = self.layer3(out)
            hiddens.append(out)

            out = self.layer4(out)
            hiddens.append(out)

            feature = self.layer5(out)
            feature = feature.view(feature.size(0), -1)
            feature = self.bn(feature)

            hiddens.append(feature)

            res = self.fc_layer(feature)

            return hiddens, res

        else:
            feature = self.feature(x)
            feature = feature.view(feature.size(0), -1)
            feature = self.bn(feature)

            res = self.fc_layer(feature)

            return feature, res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return out


class VGG16_vib(nn.Module):
    def __init__(self, n_classes, dataset='celeba'):
        super(VGG16_vib, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=True)
        self.feature = model.features
        if dataset == 'celeba':
            self.feat_dim = 512 * 2 * 2
        else:
            self.feat_dim = 512

        self.k = self.feat_dim // 2
        self.n_classes = n_classes

        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Linear(self.k, self.n_classes)

    def forward(self, x, mode="train"):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)

        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]

        std = F.softplus(std - 5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)

        return [feature, mu, std, out]



############################################ VGG ###################################################


############################################ ResNet ###################################################


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        if isinstance(x, tuple):
            x, output_list = x
        else:
            output_list = []

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)
        out = F.relu(out)
        output_list.append(out)

        return out, output_list


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, hsic_training=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(),
                                          Flatten(),
                                          nn.Linear(512 * 2 * 2, 512),
                                          nn.BatchNorm1d(512))

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

        self.hsic_training = hsic_training

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        output_list = []

        out = F.relu(self.bn1(self.conv1(x)))
        output_list.append(out)

        out, out_list = self.layer1(out)
        output_list.extend(out_list)

        out, out_list = self.layer2(out)
        output_list.extend(out_list)

        out, out_list = self.layer3(out)
        output_list.extend(out_list)

        out, out_list = self.layer4(out)
        output_list.extend(out_list)

        out = F.avg_pool2d(out, 4)  # [64, 512, 2, 2]

        out = self.output_layer(out)
        out_list.append(out)

        feature = out.view(out.size(0), -1)
        output_list.append(feature)

        res = self.fc_layer(feature)

        if self.hsic_training:
            return output_list, res
        else:
            return feature, res


def ResNet18(n_classes, hsic_training):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=n_classes, hsic_training=hsic_training)


class IR50(nn.Module):
    def __init__(self, num_classes=1000):
        super(IR50, self).__init__()
        self.feature = evolve.IR_50_64((64, 64))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(),
                                          Flatten(),
                                          nn.Linear(512 * 4 * 4, 512),
                                          nn.BatchNorm1d(512))

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

    def forward(self, x):
        feat = self.feature(x)
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        return feat, out


class IR50_vib(nn.Module):
    def __init__(self, num_classes=1000):
        super(IR50_vib, self).__init__()
        self.feature = evolve.IR_50_64((64, 64))
        self.feat_dim = 512
        self.n_classes = num_classes
        self.k = self.feat_dim // 2
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(),
                                          Flatten(),
                                          nn.Linear(512 * 4 * 4, 512),
                                          nn.BatchNorm1d(512))

        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.k, self.n_classes),
            nn.Softmax(dim=1))

    def forward(self, x):
        feat = self.output_layer(self.feature(x))
        feat = feat.view(feat.size(0), -1)
        statis = self.st_layer(feat)
        mu, std = statis[:, :self.k], statis[:, self.k:]

        std = F.softplus(std - 5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)
        __, iden = torch.max(out, dim=1)
        iden = iden.view(-1, 1)

        return feat, out, iden, mu, std


class IR152(nn.Module):
    def __init__(self, num_classes=1000):
        super(IR152, self).__init__()
        self.feature = evolve.IR_152_64((64, 64))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(),
                                          Flatten(),
                                          nn.Linear(512 * 4 * 4, 512),
                                          nn.BatchNorm1d(512))

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

    def forward(self, x):
        feat = self.feature(x)
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        res = self.fc_layer(feat)
        out = F.softmax(res, dim=1)
        __, iden = torch.max(out, dim=1)
        iden = iden.view(-1, 1)
        return feat, out, iden


class IR152_vib(nn.Module):
    def __init__(self, num_classes=1000):
        super(IR152_vib, self).__init__()
        self.feature = evolve.IR_152_64((64, 64))
        self.feat_dim = 512
        self.k = self.feat_dim // 2
        self.n_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(),
                                          Flatten(),
                                          nn.Linear(512 * 4 * 4, 512),
                                          nn.BatchNorm1d(512))

        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.k, self.n_classes),
            nn.Softmax(dim=1))

    def forward(self, x):
        feature = self.output_layer(self.feature(x))
        feature = feature.view(feature.size(0), -1)
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]

        std = F.softplus(std - 5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)
        __, iden = torch.max(out, dim=1)
        iden = iden.view(-1, 1)

        return feature, out, iden, mu, std


class ResNetCls(nn.Module):
    def __init__(self, nc=1, zdim=50, imagesize=128, nclass=8, resnetl=34, dropout=0):
        super(ResNetCls, self).__init__()
        self.backbone = ResNetL_I(resnetl, imagesize, nc)
        self.bn1 = nn.BatchNorm1d(self.backbone.final_feat_dim)
        self.fc1 = nn.Linear(self.backbone.final_feat_dim, zdim)
        self.bn2 = nn.BatchNorm1d(zdim)
        self.fc2 = nn.Linear(zdim, nclass)
        if dropout > 0:
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout = dropout

    def embed_img(self, x):
        x = self.backbone(x)
        x = F.relu(x)
        if 'dropout' in dir(self) and self.dropout > 0:
            x = self.dropout1(x)
        x = self.bn1(x)
        x = self.fc1(x)
        return x

    def embed(self, x):
        return self.embed_img(x)

    def z_to_logits(self, z):
        z = F.relu(z)
        if 'dropout' in dir(self) and self.dropout > 0:
            z = self.dropout2(z)
        z = self.bn2(z)
        z = self.fc2(z)
        return z

    def logits(self, x):
        return self.z_to_logits(self.embed(x))

    def z_to_lsm(self, z):
        z = self.z_to_logits(z)
        return F.log_softmax(z, dim=1)

    def forward(self, x):
        feature = self.embed_img(x)
        z = self.z_to_logits(feature)

        return feature, z


class PretrainedResNet(nn.Module):
    def __init__(self, nc=1, nclass=8, imagesize=128):
        super().__init__()
        # assert imagesize == 256
        self.nc = nc
        self.nclass = nclass
        self.zdim = 2048
        pretrained_imagenet_model = torchvision.models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(pretrained_imagenet_model.children())[:-1])
        self.fc = nn.Linear(self.zdim, self.nclass)

    def embed_img(self, x):
        x = self.feature_extractor(x.repeat(1, 3, 1, 1))
        x = x.reshape(x.size(0), x.size(1))
        return x

    def embed(self, x):
        return self.embed_img(x)

    def z_to_logits(self, z):
        z = self.fc(z)
        return z

    def logits(self, x):
        return self.z_to_logits(self.embed(x))

    def z_to_lsm(self, z):
        z = self.z_to_logits(z)
        return F.log_softmax(z, dim=1)

    # def forward(self, x):
    #     x = self.embed_img(x)
    #     return self.z_to_lsm(x)
    def forward(self, x):
        feature = self.embed_img(x)
        z = self.z_to_logits(feature)

        return feature, z


############################################ ResNet ###################################################


############################################ FaceNet ###################################################


class FaceNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(FaceNet, self).__init__()
        self.feature = evolve.IR_50_112((112, 112))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

    def predict(self, x):
        feat = self.feature(x)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        return out

    def forward(self, x):
        feat = self.feature(x)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        return [out]


class FaceNet64(nn.Module):
    def __init__(self, num_classes=1000):
        super(FaceNet64, self).__init__()
        self.feature = evolve.IR_50_64((64, 64))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(),
                                          Flatten(),
                                          nn.Linear(512 * 4 * 4, 512),
                                          nn.BatchNorm1d(512))

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

    def forward(self, x):
        feat = self.feature(x)
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        res = self.fc_layer(feat)
        out = F.softmax(res, dim=1)
        __, iden = torch.max(out, dim=1)
        iden = iden.view(-1, 1)
        return feat, out, iden

############################################ FaceNet ###################################################

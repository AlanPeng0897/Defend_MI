import argparse, torch, os, sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision

from torch.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np

import utils, data
from utils import mkdir
from csv_logger import CSVLogger, plot_csv
from backbone import ResNet10, ResNetL_I, ResNetL_IH
from data import normalize_image
import matplotlib.pylab as plt
from DiffAugment_pytorch import DiffAugment
from utils import save_checkpoint, maybe_load_checkpoint, to_categorical
import torchvision.utils as vutils
import chestxray
from eval_pretrained_face_classifier import PretrainedInsightFaceClassifier2, FinetunednsightFaceClassifier, \
    PretrainedInsightFaceClassifier

sys.path.append('../BiDO')
import hsic


class Net(nn.Module):
    def __init__(self, nc, nz=128, imagesize=32, dropout=0):
        super(Net, self).__init__()
        if imagesize == 32:
            self.conv1 = nn.Conv2d(nc, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)
            self.fc1 = nn.Linear(12544, nz)
            self.fc2 = nn.Linear(nz, 10)
        elif imagesize == 64:
            self.conv1 = nn.Conv2d(nc, 32, 6, 1)
            self.conv2 = nn.Conv2d(32, 64, 6, 1)
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)
            self.fc1 = nn.Linear(46656, nz)
            self.fc2 = nn.Linear(nz, 10)

    def embed_img(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

    def embed(self, x):
        return self.embed_img(x)

    def z_to_logits(self, z):
        z = F.relu(z)
        z = self.dropout2(z)
        z = self.fc2(z)
        return z

    def z_to_lsm(self, z):
        z = self.z_to_logits(z)
        return F.log_softmax(z, dim=1)

    def forward(self, x):
        x = self.embed_img(x)
        output = self.z_to_lsm(x)
        return output


class MLP(nn.Module):
    def __init__(self, nc, nz=128, imagesize=32, dropout=0):
        super(MLP, self).__init__()
        if imagesize == 32:
            self.dropout = nn.Dropout2d(dropout)
            self.fc1 = nn.Linear(32 * 32 * nc, 200)
            self.fc2 = nn.Linear(200, nz)
            self.fc3 = nn.Linear(nz, 10)
        elif imagesize == 64:
            self.dropout = nn.Dropout2d(dropout)
            self.fc1 = nn.Linear(64 * 64 * nc, 200)
            self.fc2 = nn.Linear(200, nz)
            self.fc3 = nn.Linear(nz, 10)

    def embed_img(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x

    def embed(self, x):
        return self.embed_img(x)

    def z_to_logits(self, z):
        z = F.relu(z)
        z = self.dropout(z)
        z = self.fc3(z)
        return z

    def z_to_lsm(self, z):
        z = self.z_to_logits(z)
        return F.log_softmax(z, dim=1)

    def forward(self, x):
        x = self.embed_img(x)
        output = self.z_to_lsm(x)
        return output


class Net2(nn.Module):
    def __init__(self, nc, nz):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(nc, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.dropout3 = nn.Dropout2d(0.5)
        self.bn = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, nz)
        self.fc3 = nn.Linear(nz, 10)

    def embed_img(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.sum(x, [2, 3])
        x = self.bn(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        x = self.embed_img(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


class ResNetClsH(nn.Module):
    def __init__(self, nc=3, zdim=2, imagesize=32, nclass=10, resnetl=10, dropout=0):
        super(ResNetClsH, self).__init__()
        self.backbone = ResNetL_IH(resnetl, imagesize, nc)
        self.fc1 = nn.Linear(self.backbone.final_feat_dim, zdim)
        self.bn1 = nn.BatchNorm1d(self.backbone.final_feat_dim)
        self.fc2 = nn.Linear(zdim, nclass)
        self.bn2 = nn.BatchNorm1d(zdim)
        if dropout > 0:
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout = dropout

    def embed_img(self, x, release=True):
        x, hiddens = self.backbone(x)
        x = F.relu(x)
        if 'dropout' in dir(self) and self.dropout > 0:
            x = self.dropout1(x)
        x = self.bn1(x)
        x = self.fc1(x)
        hiddens.append(x)

        if release:
            return x
        return x, hiddens

    def embed(self, x):
        return self.embed_img(x, release=False)

    def z_to_logits(self, z):
        z = F.relu(z)
        if 'dropout' in dir(self) and self.dropout > 0:
            z = self.dropout2(z)
        feat = self.bn2(z)
        z = self.fc2(feat)
        return z, feat

    def logits(self, x):
        return self.z_to_logits(self.embed(x)[0])[0]

    def z_to_lsm(self, z):
        z, feat = self.z_to_logits(z)
        return F.log_softmax(z, dim=1), feat

    def forward(self, x, release=True):
        feature, hiddens = self.embed_img(x, release=False)
        z, feat = self.z_to_lsm(feature)
        hiddens.append(feat)

        if release:
            return z
        return hiddens, z


class ResNetCls1(nn.Module):
    def __init__(self, nc=3, zdim=2, imagesize=32, nclass=10, resnetl=10, dropout=0):
        super(ResNetCls1, self).__init__()
        self.backbone = ResNetL_I(resnetl, imagesize, nc)
        self.fc1 = nn.Linear(self.backbone.final_feat_dim, zdim)
        self.bn1 = nn.BatchNorm1d(self.backbone.final_feat_dim)
        self.fc2 = nn.Linear(zdim, nclass)
        self.bn2 = nn.BatchNorm1d(zdim)
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
        x = self.embed_img(x)
        return self.z_to_lsm(x)


class ResVib(nn.Module):
    def __init__(self, nc=3, zdim=2, imagesize=32, nclass=10, resnetl=10, dropout=0):
        super(ResVib, self).__init__()
        self.backbone = ResNetL_I(resnetl, imagesize, nc)
        self.bn1 = nn.BatchNorm1d(self.backbone.final_feat_dim)
        self.fc1 = nn.Linear(self.backbone.final_feat_dim, zdim)
        # self.bn2 = nn.BatchNorm1d(zdim)
        self.feat_dim = zdim
        self.k = self.feat_dim // 2

        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Linear(self.k, nclass)

    def embed_img(self, x):
        x = self.backbone(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.fc1(x)
        return x

    def embed(self, x):
        return self.embed_img(x)

    def z_to_logits(self, feature):
        feature = feature.view(feature.size(0), -1)

        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]
        std = F.softplus(std - 5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        z = self.fc_layer(res)

        return z, mu, std

    def logits(self, x):
        return self.z_to_logits(self.embed(x))[0]

    def z_to_lsm(self, z):
        z, mu, std = self.z_to_logits(z)
        return F.log_softmax(z, dim=1), mu, std

    def forward(self, x, release=True):
        feature = self.embed_img(x)

        z, mu, std = self.z_to_lsm(feature)

        if release:
            return z

        return mu, std, z


class PretrainedResNet(nn.Module):
    def __init__(self, nc, nclass, imagesize):
        super().__init__()
        # assert imagesize == 128
        self.nc = nc
        self.nclass = nclass
        self.zdim = 2048
        pretrained_imagenet_model = torchvision.models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(pretrained_imagenet_model.children())[:-1])
        self.fc = nn.Linear(self.zdim, self.nclass)

    def embed_img(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = self.feature_extractor(x)
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

    def forward(self, x):
        x = self.embed_img(x)
        return self.z_to_lsm(x)


class CelebAPretrained:
    def __init__(self):
        self.pretrained_net = PretrainedInsightFaceClassifier2('cuda', tta=False)

    def embed_img(self, x):
        return self.pretrained_net.embed(x)

    def embed(self, x):
        return self.embed_img(x)

    def z_to_logits(self, z):
        return self.pretrained_net.z_to_logits(z)
        # return self.pretrained_net.z_to_logits(z)

    def logits(self, x):
        return self.z_to_logits(self.embed(x))

    def z_to_lsm(self, z):
        z = self.z_to_logits(z)
        return F.log_softmax(z, dim=1)

    def forward(self, x):
        x = self.embed_img(x)
        return self.z_to_lsm(x)


class VGG(ResNetCls1):
    def __init__(self, zdim=2, nclass=10, dropout=0):
        super(VGG, self).__init__(nc=3, zdim=zdim, imagesize=32,
                                  nclass=nclass, resnetl=10, dropout=dropout)
        self.backbone = torchvision.models.vgg16_bn()
        self.fc1 = nn.Linear(1000, zdim)
        self.bn1 = nn.BatchNorm1d(1000)


class ResNetCls(nn.Module):
    def __init__(self, nc=3):
        super(ResNetCls, self).__init__()
        self.backbone = ResNet10(nc=nc)
        self.fc2 = nn.Linear(self.backbone.final_feat_dim, 10)

    def embed_img(self, x):
        x = self.backbone(x)
        return x

    def z_to_lsm(self, z):
        z = F.relu(z)
        z = self.fc2(z)
        return F.log_softmax(z, dim=1)

    def forward(self, x):
        x = self.embed_img(x)
        return self.z_to_lsm(x)


def train_H(args, model, device, train_loader, optimizer, epoch, iteration_logger, n_classes, a1, a2):
    model.train()
    for batch_idx, (x, target) in enumerate(train_loader):
        loss = 0

        hx_l_list = []
        hy_l_list = []

        x, target = x.to(device), target.to(device)
        x = DiffAugment(x / 2 + .5, args.augment).clamp(0, 1) * 2 - 1
        optimizer.zero_grad()

        hiddens, output = model(x, release=False)
        CE = F.nll_loss(output, target)
        loss += CE

        bs = x.size(0)
        h_target = to_categorical(target, num_classes=n_classes).float()
        h_data = x.view(bs, -1)

        for hidden in hiddens:
            hidden = hidden.view(bs, -1)
            if args.measure == 'HSIC':
                hxz_l, hyz_l = hsic.hsic_objective(
                    hidden,
                    h_target=h_target.float(),
                    h_data=h_data,
                    sigma=5.,
                    ktype='linear'
                )
            elif args.measure == 'COCO':
                hxz_l, hyz_l = hsic.coco_objective(
                    hidden,
                    h_target=h_target.float(),
                    h_data=h_data,
                    sigma=5.,
                    ktype='linear'
                )
            temp_hsic = a1 * hxz_l - a2 * hyz_l
            loss += temp_hsic
            hx_l_list.append(round(hxz_l.item(), 2))
            hy_l_list.append(round(hyz_l.item(), 2))

        loss.backward()
        optimizer.step()

        lxz = sum(hx_l_list) / len(hx_l_list)
        lyz = sum(hy_l_list) / len(hy_l_list)
        # Acc
        acc = (output.max(-1)[1] == target).float().mean().item()

        # Log
        if batch_idx % args.log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)] | CE:{:.4f} | Loss: {:.4f} | Lxz(down):{:.4f} | Lyz(up):{:.4f} | Acc: {:.4f}'.format(
                    epoch, batch_idx * len(x), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), CE.item(), loss.item(), lxz, lyz, acc))

            iteration_logger.writerow({
                'global_iteration': batch_idx + len(train_loader) * epoch,
                'train_acc': acc,
                'train_loss': loss.item(),
                'lxz': lxz,
                'lyz': lyz
            })
            plot_csv(iteration_logger.filename, os.path.join(args.output_dir, 'iteration_plots.jpeg'))

    # Sanity check: vis data
    print(hx_l_list)
    print(hy_l_list)
    if epoch == 1:
        vutils.save_image(x[:64], '%s/train_batch.jpeg' %
                          args.output_dir, normalize=True, nrow=8)


def train_V(args, model, device, train_loader, optimizer, epoch, iteration_logger, beta):
    model.train()
    for batch_idx, (x, target) in enumerate(train_loader):
        x, target = x.to(device), target.to(device)
        x = DiffAugment(x / 2 + .5, args.augment).clamp(0, 1) * 2 - 1
        optimizer.zero_grad()
        mu, std, output = model(x, release=False)
        info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean()
        loss = F.nll_loss(output, target) + beta * info_loss
        # loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()

        # Acc
        acc = (output.max(-1)[1] == target).float().mean().item()

        # Log
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.4f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), acc))

            iteration_logger.writerow({
                'global_iteration': batch_idx + len(train_loader) * epoch,
                'train_acc': acc,
                'train_loss': loss.item(),
            })
            plot_csv(iteration_logger.filename, os.path.join(
                args.output_dir, 'iteration_plots.jpeg'))

    # Sanity check: vis data
    if epoch == 1:
        vutils.save_image(x[:64], '%s/train_batch.jpeg' %
                          args.output_dir, normalize=True, nrow=8)


def train(args, model, device, train_loader, optimizer, epoch, iteration_logger):
    model.train()
    for batch_idx, (x, target) in enumerate(train_loader):
        x, target = x.to(device), target.to(device)
        x = DiffAugment(x / 2 + .5, args.augment).clamp(0, 1) * 2 - 1
        optimizer.zero_grad()
        output = model(x)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # Acc
        acc = (output.max(-1)[1] == target).float().mean().item()

        # Log
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.4f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), acc))

            iteration_logger.writerow({
                'global_iteration': batch_idx + len(train_loader) * epoch,
                'train_acc': acc,
                'train_loss': loss.item(),
            })
            plot_csv(iteration_logger.filename, os.path.join(
                args.output_dir, 'iteration_plots.jpeg'))

    # Sanity check: vis data
    if epoch == 1:
        vutils.save_image(x[:64], '%s/train_batch.jpeg' %
                          args.output_dir, normalize=True, nrow=8)


def test(args, model, device, test_loader, epoch=-1, plot_embed=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x, target in test_loader:
            x, target = x.to(device), target.to(device)

            if args.model == 'ResNetClsH':
                hiddens, output = model(x, release=False)
            else:
                output = model(x)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    # Sanity check: vis data
    if epoch == 1:
        vutils.save_image(x[:64], '%s/test_batch.jpeg' % args.output_dir, normalize=True, nrow=8)

    if plot_embed and isinstance(model, ResNetCls1):
        e = model.embed_img(x).detach().cpu().numpy()
        plt.clf()
        for c in range(990, 1000):
            x, y = e[target.cpu().numpy() == c][:, 0], e[target.cpu().numpy() == c][:, 1]
            plt.scatter(x, y, label=f"{c}")
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, f'embed_{epoch}.jpeg'))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


def get_model(args, device):
    # Model
    if args.dataset == 'mnist':
        if args.model == 'ResNetCls1':
            model = ResNetCls1(1, zdim=args.latent_dim,
                               imagesize=args.imageSize).to(device)
            optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
            scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
            args.epochs = 13
        elif args.model == 'Net2':
            # model = Net2(1, args.latent_dim).to(device)
            raise
        elif args.model == 'Net':
            model = Net(1, args.latent_dim,
                        imagesize=args.imageSize).to(device)
            optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
            scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
            args.epochs = 13
        elif args.model == 'MLP':
            model = MLP(1, args.latent_dim,
                        imagesize=args.imageSize).to(device)
            optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
            scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
            args.epochs = 13
        elif args.model == 'ResNetClsH':
            model = ResNetClsH(1, zdim=args.latent_dim).to(device)
            optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
            scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
            args.epochs = 13
        elif args.model == 'ResVib':
            model = ResVib(1, zdim=args.latent_dim).to(device)
            optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
            scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
            args.epochs = 13
        else:
            raise ValueError()

    if args.dataset == 'omniglot':
        if args.model == 'ResNetCls1':
            model = ResNetCls1(1, zdim=args.latent_dim,
                               imagesize=args.imageSize, nclass=964).to(device)
            optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
            scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
            args.epochs = 13
        else:
            raise ValueError()

    elif args.dataset == 'mnist-omniglot':
        assert args.model == 'ResNetCls1'
        model = ResNetCls1(1, zdim=args.latent_dim,
                           imagesize=args.imageSize, nclass=11).to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        args.epochs = 13

    elif args.dataset in ['cub-train', 'cub', 'cifar-fs-train']:
        assert args.model == 'ResNetCls1'
        nclass = {
            'cub-train': 100,
            'cub': 200,
            'cifar-fs-train': 64
        }[args.dataset]
        model = ResNetCls1(3, zdim=args.latent_dim,
                           imagesize=args.imageSize, nclass=nclass).to(device)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3, nesterov=True)

        if args.epochs == 200:
            scheduler = MultiStepLR(optimizer, milestones=[
                60, 120, 160], gamma=0.2)
        elif args.epochs == 500:
            scheduler = MultiStepLR(optimizer, milestones=[
                150, 300, 400], gamma=0.2)
        else:
            raise

    elif args.dataset in ['cifar10', 'cifar0to4', 'svhn']:
        if args.model == 'ResNetCls1':
            model = ResNetCls1(3, zdim=args.latent_dim).to(device)
        elif args.model == 'ResNetCls':
            model = ResNetCls().to(device)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = MultiStepLR(optimizer, milestones=[
            60, 120, 160], gamma=0.2)
        args.epochs = 200

    elif args.dataset.startswith('celeba') or args.dataset in ['pubfig83', 'cfw']:
        if args.dataset.startswith('celeba'):
            nclass = 1000
        elif args.dataset == 'pubfig83':
            nclass = 83
        elif args.dataset == 'cfw':
            nclass = 100

        if args.model == 'PretrainedInsightFaceClassifier':
            model = PretrainedInsightFaceClassifier('cuda', pad=True)
            return model, None, None

        if args.model == 'ResNetCls1':
            model = ResNetCls1(3, zdim=args.latent_dim, imagesize=64, nclass=nclass,
                               resnetl=args.resnetl, dropout=args.dropout).to(device)
        elif args.model == 'vgg':
            model = VGG(zdim=args.latent_dim, nclass=nclass,
                        dropout=args.dropout).to(device)

        elif args.model == 'ResNetClsH':
            model = ResNetClsH(3, zdim=args.latent_dim, imagesize=64, nclass=nclass,
                               resnetl=args.resnetl, dropout=args.dropout).to(device)
        elif args.model == 'ResVib':
            model = ResVib(3, zdim=args.latent_dim, imagesize=64, nclass=nclass,
                           resnetl=args.resnetl, dropout=args.dropout).to(device)
        else:
            raise

        if args.model == 'ResNetClsH' or args.model == 'ResVib':
            args.epochs = 100
            # optimizer = torch.optim.Adam(model.parameters(), 5e-4)
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
            scheduler = MultiStepLR(optimizer, milestones=[60, 80], gamma=0.2)
        else:
            args.epochs = 200
            optimizer = torch.optim.SGD(
                model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

            if args.epochs == 200:
                scheduler = MultiStepLR(optimizer, milestones=[
                    60, 120, 160], gamma=0.2)
            elif args.epochs == 2000:
                scheduler = MultiStepLR(optimizer, milestones=[
                    600, 1200, 1600], gamma=0.2)

    elif args.dataset == 'chestxray':
        nclass = 8
        if args.model == 'ResNetCls1':
            model = ResNetCls1(1, zdim=args.latent_dim, imagesize=args.imageSize, nclass=nclass,
                               resnetl=args.resnetl, dropout=args.dropout).to(device)
        elif args.model == 'PretrainedResNet':
            model = PretrainedResNet(1, nclass=nclass, imagesize=args.imageSize).to(device)
        elif args.model == 'ResNetClsH':
            model = ResNetClsH(1, zdim=args.latent_dim, imagesize=128, nclass=nclass,
                               resnetl=args.resnetl, dropout=args.dropout).to(device)
        elif args.model == 'ResVib':
            model = ResVib(1, zdim=args.latent_dim, imagesize=128, nclass=nclass,
                               resnetl=args.resnetl, dropout=args.dropout).to(device)
        else:
            raise

        # Same as CelebA
        if args.model == 'ResNetClsH' or args.model == 'ResVib':
            args.epochs = 100

            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
            scheduler = MultiStepLR(optimizer, milestones=[60, 80], gamma=0.2)
        else:
            args.epochs = 200
            optimizer = torch.optim.SGD(
                model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
            if args.epochs == 200:
                scheduler = MultiStepLR(optimizer, milestones=[
                    60, 120, 160], gamma=0.2)
            elif args.epochs == 300:
                scheduler = MultiStepLR(optimizer, milestones=[
                    120, 240, 270], gamma=0.2)
            elif args.epochs == 2000:
                scheduler = MultiStepLR(optimizer, milestones=[
                    600, 1200, 1600], gamma=0.2)

    # model = torch.nn.DataParallel(model).to(device)
    return model, optimizer, scheduler


def main(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Logging
    epoch_fieldnames = ['global_iteration', 'test_acc']
    epoch_logger = CSVLogger(every=1,
                             fieldnames=epoch_fieldnames,
                             filename=os.path.join(
                                 args.output_dir, 'epoch_log.csv'),
                             resume=args.resume)

    iteration_fieldnames = ['global_iteration', 'train_acc', 'train_loss', 'lxz', 'lyz']
    iteration_logger = CSVLogger(every=1,
                                 fieldnames=iteration_fieldnames,
                                 filename=os.path.join(
                                     args.output_dir, 'iteration_log.csv'),
                                 resume=args.resume)

    # Data
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    if args.dataset == 'mnist':
        dat = data.load_data(args.dataset, args.dataroot,
                             device=device, imgsize=args.imageSize, Ntrain=args.Ntrain, Ntest=args.Ntest,
                             dataset_size=args.dataset_size)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                normalize_image(dat['X_train'] / 2 + 0.5, args.dataset), dat['Y_train']),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                normalize_image(dat['X_test'] / 2 + 0.5, args.dataset), dat['Y_test']),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)

    elif args.dataset in ['pubfig83', 'cfw']:
        dat = data.load_data(args.dataset, args.dataroot,
                             device=device, imgsize=args.imageSize, Ntrain=args.Ntrain, Ntest=args.Ntest,
                             dataset_size=args.dataset_size)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(dat['X_train'], dat['Y_train']),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(dat['X_test'], dat['Y_test']),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)

    elif args.dataset == 'omniglot':
        dat = data.load_data('omniglot', args.dataroot,
                             device=device, imgsize=args.imageSize)

        # Split the examples in train classes by examples
        Ntrain = 15
        Nclass = len(set(dat['Y_train'].numpy()))

        Xtrain = []
        Ytrain = []
        Xtest = []
        Ytest = []
        for c in range(Nclass):
            idx = dat['Y_train'] == c
            xc = dat['X_train'][idx]
            yc = dat['Y_train'][idx]
            Xtrain.append(xc[:Ntrain])
            Xtest.append(xc[Ntrain:])
            Ytrain.append(yc[:Ntrain])
            Ytest.append(yc[Ntrain:])
        Xtrain = torch.cat(Xtrain)
        Ytrain = torch.cat(Ytrain)
        Xtest = torch.cat(Xtest)
        Ytest = torch.cat(Ytest)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xtrain, Ytrain),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xtest, Ytest),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)

    elif args.dataset == 'cifar-fs-train':
        dat = data.load_data('cifar-fs-train', args.dataroot,
                             device=device, imgsize=args.imageSize)

        # Split the examples in train classes by examples
        Ntrain = 500
        Nclass = len(set(dat['Y_train'].numpy()))

        Xtrain = []
        Ytrain = []
        Xtest = []
        Ytest = []
        for c in range(Nclass):
            idx = dat['Y_train'] == c
            xc = dat['X_train'][idx]
            yc = dat['Y_train'][idx]
            Xtrain.append(xc[:Ntrain])
            Xtest.append(xc[Ntrain:])
            Ytrain.append(yc[:Ntrain])
            Ytest.append(yc[Ntrain:])
        Xtrain = torch.cat(Xtrain)
        Ytrain = torch.cat(Ytrain)
        Xtest = torch.cat(Xtest)
        Ytest = torch.cat(Ytest)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xtrain, Ytrain),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xtest, Ytest),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)

    elif args.dataset in ['cub', 'cub-train']:
        dat = data.load_data('cub-train', args.dataroot,
                             device=device, imgsize=args.imageSize)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(dat['X_train'], dat['Y_train']),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(dat['X_test'], dat['Y_test']),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # Cifar datasets can't use my data loader because it
    # needs data augmentation (without it leads to >5pt acc drop)
    elif args.dataset == 'cifar10':
        # Data
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR10(
            root=os.path.join(args.dataroot), train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        testset = datasets.CIFAR10(
            root=os.path.join(args.dataroot), train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    elif args.dataset == 'cifar0to4':
        # Data
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4907, 0.4856, 0.4509),
                                 (0.2454, 0.2415, 0.2620)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4907, 0.4856, 0.4509),
                                 (0.2454, 0.2415, 0.2620)),
        ])

        trainset = datasets.CIFAR10(
            root=os.path.join(args.dataroot), train=True, download=True, transform=transform_train)
        idxs = np.array(trainset.targets) <= 4
        trainset.data = trainset.data[idxs]
        trainset.targets = np.array(trainset.targets)[idxs]
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        testset = datasets.CIFAR10(
            root=os.path.join(args.dataroot), train=False, download=True, transform=transform_test)
        idxs = np.array(testset.targets) <= 4
        testset.data = testset.data[idxs]
        testset.targets = np.array(testset.targets)[idxs]
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    elif args.dataset == 'svhn':
        # Data
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
        ])

        trainset = datasets.SVHN(
            root=os.path.join(args.dataroot), split='train', download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        testset = datasets.SVHN(
            root=os.path.join(args.dataroot), split='test', download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    elif args.dataset.startswith('celeba'):
        import celeba
        train_x, train_y, test_x, test_y = celeba.get_celeba_dataset(
            'target', crop='crop' in args.dataset)
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            train_x, train_y), batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            test_x, test_y), batch_size=args.batch_size, shuffle=False, num_workers=2)

    elif args.dataset == 'chestxray':
        train_x, train_y, test_x, test_y = chestxray.load_data_cache(args.imageSize)
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            train_x, train_y), batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            test_x, test_y), batch_size=args.batch_size, shuffle=False, num_workers=2)

    model, optimizer, scheduler = get_model(args, device)

    if args.eval:
        # Load model and verify accuracy
        path = os.path.join(args.output_dir, "best_ckpt.pt")
        if args.model == 'PretrainedInsightFaceClassifier':
            for x, y in test_loader:
                test_acc = model.acc(x, y)
                print(f"Test ACC: {test_acc}")
            sys.exit(0)

        model.load_state_dict(torch.load(path))
        model.eval()
        test_acc = test(args, model, device, test_loader)
        print(f"Test ACC: {test_acc}")

        sys.exit(0)

    # Check for ckpt
    ckpt = maybe_load_checkpoint(args)

    if ckpt is not None:
        start_epoch = ckpt['epoch']
        optimizer.load_state_dict(ckpt['optimizer'])
        model.load_state_dict(ckpt['model'])
    else:
        start_epoch = 1

    best_test_acc = -1

    for epoch in range(start_epoch, args.epochs + 1):
        # Ckpt
        state = {
            "optimizer": optimizer.state_dict(),
            "model": model.state_dict(),
            "epoch": epoch,
        }
        save_checkpoint(args, state)

        print("lr: ", optimizer.param_groups[0]['lr'])
        print("batch_size: ", args.batch_size)
        if args.model == 'ResNetClsH':
            a1, a2 = args.a1, args.a2
            print('a1:', a1, 'a2:', a2)
            if args.dataset.startswith('celeba'):
                n_classes = 1000
            elif args.dataset == 'mnist':
                n_classes = 10
            else:
                n_classes = 8

            train_H(args, model, device, train_loader, optimizer, epoch, iteration_logger, n_classes, a1, a2)
            test_acc = test(args, model, device, test_loader, epoch=epoch)
            scheduler.step()

        elif args.model == 'ResVib':
            beta = args.beta
            train_V(args, model, device, train_loader, optimizer, epoch, iteration_logger, beta)
            test_acc = test(args, model, device, test_loader, epoch=epoch)
            scheduler.step()

        else:
            train(args, model, device, train_loader, optimizer, epoch, iteration_logger)
            test_acc = test(args, model, device, test_loader, epoch=epoch)
            scheduler.step()

        epoch_logger.writerow({
            'global_iteration': epoch,
            'test_acc': test_acc,
        })
        plot_csv(epoch_logger.filename, os.path.join(
            args.output_dir, 'epoch_plots.jpeg'))

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_ckpt.pt"))

    print(f"best_test_acc: {best_test_acc}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--imageSize', type=int, default=32,
                        help='the height / width of the input image to network')
    parser.add_argument('--Ntrain', type=int, default=60000,
                        help='training set size for stackedmnist')
    parser.add_argument('--Ntest', type=int, default=10000,
                        help='test set size for stackedmnist')
    parser.add_argument('--dataset_size', type=int, default=-1)

    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--dataroot', default='data')
    parser.add_argument('--dataset', required=False, default="celeba",
                        choices=['mnist', 'cifar10', 'cifar0to4',
                                 'celeba', 'celeba_crop', 'omniglot', 'cifar-fs-train', 'cub-train', 'cub', 'pubfig83',
                                 'cfw', 'chestxray'])
    parser.add_argument('--output_dir', required=False, help='', default='./clf_results/celeba/vib/')
    parser.add_argument('--eval', type=int, default=0)
    parser.add_argument('--latent_dim', type=int, default=50)
    parser.add_argument('--model', type=str, required=False, default="ResVib",
                        choices=['PretrainedResNet', 'vgg', 'ResNetCls1', 'ResNetClsH',
                                 'PretrainedInsightFaceClassifier', 'ResVib'])
    parser.add_argument('--a1', type=float, default=0.1)
    parser.add_argument('--a2', type=float, default=5)
    parser.add_argument('--beta', type=float, default=0)
    parser.add_argument('--resnetl', type=int, default=34, choices=[10, 34, 50])
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--augment', nargs='?', const='', type=str, default='cutout',
                        choices=['', 'cutout', 'translation', 'color', 'translation12'])
    parser.add_argument('--resume', type=int, required=False, default=0)
    parser.add_argument('--user', type=str, default='wangkuan')
    parser.add_argument('--measure', type=str, default='HSIC')
    parser.add_argument('--patience', type=int, default=50)

    args = parser.parse_args()

    args.ckpt = os.path.join(args.output_dir, "ckpt.pt")
    # args.output_dir = os.path.join('./clf_results', args.dataset, args.measure, f'{args.a1}&{args.a2}')
    '''
     python classify_mnist.py --imageSize=128 --epochs=100 --dataset=chestxray --output_dir='./clf_results/chestxray/hsic_0.05&0.5' --a1=0.05 --a2=0.5 --augment=translation
    '''
    mkdir(args.output_dir)
    args.jobid = os.environ['SLURM_JOB_ID'] if 'SLURM_JOB_ID' in os.environ else -1
    utils.save_args(args, os.path.join(args.output_dir, f'args.json'))

    main(args)

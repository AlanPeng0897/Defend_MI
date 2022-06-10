import numpy as np
import torch, random, sys, json, time, dataloader, copy, os
import torch.nn as nn
from datetime import datetime
from torch.utils.data import sampler
from collections import defaultdict
from torch.autograd import Variable
import torchvision.utils as tvls


def save_tensor_images(images, filename, nrow = None, normalize = True):
    if not nrow:
        tvls.save_image(images, filename, normalize = normalize, padding=0)
    else:
        tvls.save_image(images, filename, normalize = normalize, nrow=nrow, padding=0)

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

def load_my_state_dict(self, state_dict):
    own_state = self.state_dict()
    # print(state_dict)
    for name, param in state_dict.items():
        if name not in own_state:
            print(name)
            continue
        # print(param.data.shape)
        own_state[name].copy_(param.data)

def load_state_dict(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)


def load_peng_state_dict(net, state_dict):
    print("load self-constructed model!!!")
    net_state = net.state_dict()
    for ((name, param), (old_name, old_param),) in zip(net_state.items(), state_dict.items()):
        # print(name, '---', old_name)
        net_state[name].copy_(old_param.data)


def load_pretrain(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name.startswith("module.fc_layer"):
            continue
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)


def load_params(self, model):
    own_state = self.state_dict()
    for name, param in model.named_parameters():
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)


def load_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data


def print_params(info, params, dataset=None):
    print('-----------------------------------------------------------------')
    if dataset is not None:
        print("Dataset: %s" % dataset)
        print("Running time: %s" % datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    for i, (key, value) in enumerate(info.items()):
        if i >= 3:
            print("%s: %s" % (key, str(value)))
    for i, (key, value) in enumerate(params.items()):
        print("%s: %s" % (key, str(value)))
    print('-----------------------------------------------------------------')


def init_dataloader(args, file_path, mode="gan"):
    tf = time.time()
    model_name = args['dataset']['model_name']
    bs = args[model_name]['batch_size']
    if args['dataset']['name'] == "celeba" or args['dataset']['name'] == "facescrub" \
            or args['dataset']['name'] == "cifar":
        data_set = dataloader.ImageFolder(args, file_path, mode)
    elif args['dataset']['name'] == "mnist" or args['dataset']['name'] == "chestxray":
        data_set = dataloader.GrayFolder(args, file_path, mode)

    if mode == "train":
        if args['dataset']['name'] == "celeba":
            sampler = RandomIdentitySampler(data_set, args[model_name]['batch_size'], args['dataset']['instance'])
            data_loader = torch.utils.data.DataLoader(data_set,
                                                      sampler=sampler,
                                                      batch_size=bs,
                                                      num_workers=args['dataset']['num_workers'],
                                                      pin_memory=True,
                                                      drop_last=True)
        else:
            data_loader = torch.utils.data.DataLoader(data_set,
                                                      shuffle=True,
                                                      batch_size=bs,
                                                      num_workers=args['dataset']['num_workers'],
                                                      pin_memory=True,
                                                      drop_last=True)
    else:
        data_loader = torch.utils.data.DataLoader(data_set,
                                                  shuffle=False,
                                                  batch_size=bs,
                                                  num_workers=args['dataset']['num_workers'],
                                                  pin_memory=True,
                                                  drop_last=True)

    interval = time.time() - tf
    print('Initializing data loader took {:.2f}'.format(interval))
    return data_loader


class RandomIdentitySampler(sampler.Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    """

    def __init__(self, dataset, batch_size, num_instances):
        self.data_source = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        # changed according to the dataset
        for index, inputs in enumerate(self.data_source):
            self.index_dic[inputs[1]].append(index)

        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


def weights_init_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


from torchvision import transforms


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


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.squeeze(torch.eye(num_classes)[y], dim=1)


from util import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


def train_vib(trainloader, model, criterion, optimizer, beta=1e-2):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        _, mu, std, out_prob = model(inputs)
        cross_loss = criterion(out_prob, targets)
        info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean()
        loss = cross_loss + beta * info_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(out_prob.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    return losses.avg, top1.avg


def test_vib(testloader, model, criterion, beta=1e-2):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            _, mu, std, out_prob = model(inputs)
            cross_loss = criterion(out_prob, targets)
            info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean()
            loss = cross_loss + beta * info_loss

            # measure accuracy and record loss
            prec1, prec5 = accuracy(out_prob.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
    bar.finish()
    return losses.avg, top1.avg


def train(trainloader, model, criterion, optimizer):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        _, outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        # -----------------------------------------
        # for k, m in enumerate(model.modules()):
        #     # print(k, m)
        #     if isinstance(m, nn.Conv2d):
        #         weight_copy = m.weight.data.abs().clone()
        #         mask = weight_copy.gt(0).float().cuda()
        #         m.weight.grad.data.mul_(mask)
        # -----------------------------------------
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    return losses.avg, top1.avg


def test(testloader, model, criterion):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            _, outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
    bar.finish()
    return losses.avg, top1.avg


def save_checkpoint(state, directory, filename):
    filepath = os.path.join(directory, filename)
    torch.save(state, filepath)

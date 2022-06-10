import torch, os, time, model, utils, hsic, sys
import numpy as np
import torch.nn as nn
from copy import deepcopy
from torch.optim.lr_scheduler import MultiStepLR
from util import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import torch.nn.functional as F
from tqdm import tqdm

device = "cuda"


def test(model, criterion, dataloader):
    tf = time.time()
    model.eval()
    loss, cnt, ACC = 0.0, 0, 0

    for img, iden in dataloader:
        img, iden = img.to(device), iden.to(device)
        bs = img.size(0)
        iden = iden.view(-1)

        out_digit = model(img)[-1]
        out_iden = torch.argmax(out_digit, dim=1).view(-1)
        ACC += torch.sum(iden == out_iden).item()
        cnt += bs

    return ACC * 100.0 / cnt


def multilayer_hsic(model, criterion, inputs, target, a1, a2, n_classes, ktype, hsic_training, measure):
    hx_l_list = []
    hy_l_list = []
    bs = inputs.size(0)
    total_loss = 0

    if hsic_training:
        hiddens, out_digit = model(inputs)
        cross_loss = criterion(out_digit, target)

        total_loss += cross_loss
        h_target = utils.to_categorical(target, num_classes=n_classes).float()
        h_data = inputs.view(bs, -1)
        for hidden in hiddens:
            hidden = hidden.view(bs, -1)

            if measure == 'HSIC':
                hxz_l, hyz_l = hsic.hsic_objective(
                    hidden,
                    h_target=h_target.float(),
                    h_data=h_data,
                    sigma=5.,
                    ktype=ktype
                )
            elif measure == 'COCO':
                hxz_l, hyz_l = hsic.coco_objective(
                    hidden,
                    h_target=h_target.float(),
                    h_data=h_data,
                    sigma=5.,
                    ktype=ktype
                )

            temp_hsic = a1 * hxz_l - a2 * hyz_l
            total_loss += temp_hsic

            hx_l_list.append(round(hxz_l.item(), 5))
            hy_l_list.append(round(hyz_l.item(), 5))

    else:
        feats, out_digit = model(inputs)
        cross_loss = criterion(out_digit, target)

        total_loss += cross_loss
        h_target = utils.to_categorical(target, num_classes=n_classes).float()
        h_data = inputs.view(bs, -1)
        # hidden = feats.view(bs, -1)

        # hxy_l = hsic.hsic_normalized_cca(h_data, out_digit, sigma=5., ktype=ktype)
        hxy_l = hsic.coco_normalized_cca(h_data, out_digit, sigma=5., ktype=ktype)

        # hxz_l, hyz_l = hsic.hsic_objective(
        #     hidden,
        #     h_target=h_target,
        #     h_data=h_data,
        #     sigma=5.,
        #     ktype=ktype
        # )

        # temp_hsic = a1 * hxz_l - a2 * hyz_l
        temp_hsic = a1 * hxy_l
        total_loss += temp_hsic

        hxz_l = hxy_l
        hyz_l = hxy_l
        hx_l_list.append(round(hxz_l.item(), 5))
        hy_l_list.append(round(hyz_l.item(), 5))

    return total_loss, cross_loss, out_digit, hx_l_list, hy_l_list


def train_HSIC(model, criterion, optimizer, trainloader, a1, a2, n_classes,
               ktype='gaussian', hsic_training=True,measure='HSIC'):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_cls = AverageMeter()
    lxz, lyz = AverageMeter(), AverageMeter()
    end = time.time()

    pbar = tqdm(enumerate(trainloader), total=len(trainloader), ncols=150)

    for batch_idx, (inputs, iden) in pbar:
        data_time.update(time.time() - end)
        bs = inputs.size(0)
        inputs, iden = inputs.to(device), iden.to(device)
        iden = iden.view(-1)

        loss, cross_loss, out_digit, hx_l_list, hy_l_list = multilayer_hsic(model, criterion, inputs, iden, a1, a2,
                                                                            n_classes, ktype, hsic_training, measure)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(out_digit.data, iden.data, topk=(1, 5))
        losses.update(loss.item())
        loss_cls.update(float(cross_loss.detach().cpu().numpy()))
        lxz.update(sum(hx_l_list) / len(hx_l_list))
        lyz.update(sum(hy_l_list) / len(hy_l_list))

        top1.update(prec1.item())
        top5.update(prec5.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        msg = 'CE:{cls:.4f} | Lxz(down):{lxz:.5f} | Lyz(up):{lyz:.5f} | Loss:{loss:.4f} | ' \
              'top1:{top1: .4f} | top5:{top5: .4f}'.format(
            cls=loss_cls.avg,
            lxz=lxz.avg,
            lyz=lyz.avg,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        pbar.set_description(msg)
    print("hx_l_list:", hx_l_list)
    print("hy_l_list:", hy_l_list)
    return losses.avg, top1.avg


def test_HSIC(model, criterion, testloader, a1, a2, n_classes, ktype='gaussian', hsic_training=True,measure='HSIC'):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_cls = AverageMeter()
    lxz, lyz = AverageMeter(), AverageMeter()
    end = time.time()

    pbar = tqdm(enumerate(testloader), total=len(testloader), ncols=150)
    with torch.no_grad():
        for batch_idx, (inputs, iden) in pbar:
            data_time.update(time.time() - end)

            inputs, iden = inputs.to(device), iden.to(device)
            bs = inputs.size(0)
            iden = iden.view(-1)

            loss, cross_loss, out_digit, hx_l_list, hy_l_list = multilayer_hsic(model, criterion, inputs, iden, a1, a2,
                                                                                n_classes, ktype, hsic_training, measure)

            # measure accuracy and record loss

            prec1, prec5 = accuracy(out_digit.data, iden.data, topk=(1, 5))
            losses.update(loss.item(), bs)
            loss_cls.update(cross_loss.item(), bs)
            lxz.update(sum(hx_l_list) / len(hx_l_list), bs)
            lyz.update(sum(hy_l_list) / len(hy_l_list), bs)

            top1.update(prec1.item(), bs)
            top5.update(prec5.item(), bs)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            msg = 'CE:{cls:.4f} | Lxz(down):{lxz:.5f} | Lyz(up):{lyz:.5f} | Loss:{loss:.4f} | ' \
                  'top1:{top1: .4f} | top5:{top5: .4f}'.format(
                cls=loss_cls.avg,
                lxz=lxz.avg,
                lyz=lyz.avg,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            pbar.set_description(msg)

    print("hx_l_list:", hx_l_list)
    print("hy_l_list:", hy_l_list)
    print('-' * 80)
    return losses.avg, top1.avg


def train_reg(model, criterion, optimizer, trainloader):
    model.train()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_cls = AverageMeter()
    end = time.time()

    pbar = tqdm(enumerate(trainloader), total=len(trainloader), ncols=150)

    for batch_idx, (inputs, iden) in pbar:
        inputs, iden = inputs.to(device), iden.to(device)
        iden = iden.view(-1)

        feats, out_digit = model(inputs)
        cross_loss = criterion(out_digit, iden)
        # triplet_loss = triplet(feats, iden)
        loss = cross_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        bs = inputs.size(0)
        prec1, prec5 = accuracy(out_digit.data, iden.data, topk=(1, 5))
        losses.update(loss.item(), bs)
        loss_cls.update(cross_loss.item(), bs)

        top1.update(prec1.item(), bs)
        top5.update(prec5.item(), bs)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        msg = '({batch}/{size}) | ' \
              'Loss:{loss:.4f} | ' \
              'top1:{top1: .4f} | top5:{top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            cls=loss_cls.avg,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        pbar.set_description(msg)

    return losses.avg, top1.avg


def test_reg(model, criterion, testloader):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    pbar = tqdm(enumerate(testloader), total=len(testloader), ncols=150)

    with torch.no_grad():
        for batch_idx, (inputs, iden) in pbar:
            data_time.update(time.time() - end)

            inputs, iden = inputs.to(device), iden.to(device)
            bs = inputs.size(0)
            iden = iden.view(-1)
            feats, out_digit = model(inputs)
            cross_loss = criterion(out_digit, iden)

            loss = cross_loss

            # measure accuracy and record loss
            prec1, prec5 = accuracy(out_digit.data, iden.data, topk=(1, 5))
            losses.update(loss.item(), bs)
            top1.update(prec1.item(), bs)
            top5.update(prec5.item(), bs)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            # plot progress
            msg = '({batch}/{size}) | ' \
                  'Loss:{loss:.4f} | ' \
                  'top1:{top1: .4f} | top5:{top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            pbar.set_description(msg)

    return losses.avg, top1.avg


def train_vib(model, criterion, optimizer, trainloader, beta=1e-2):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    pbar = tqdm(enumerate(trainloader), total=len(trainloader), ncols=150)

    for batch_idx, (inputs, targets) in pbar:
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.cuda(), targets.cuda()
        bs = inputs.size(0)

        # compute output
        _, mu, std, out_digit = model(inputs)
        cross_loss = criterion(out_digit, targets)
        info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean()
        loss = cross_loss + beta * info_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(out_digit.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), bs)
        top1.update(prec1.item(), bs)
        top5.update(prec5.item(), bs)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        # plot progress
        msg = '({batch}/{size}) | ' \
              'Loss:{loss:.4f} | ' \
              'top1:{top1: .4f} | top5:{top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            cls=losses.avg,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        pbar.set_description(msg)
    return losses.avg, top1.avg


def test_vib(model, criterion, testloader, beta=1e-2):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    pbar = tqdm(enumerate(testloader), total=len(testloader), ncols=150)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in pbar:
            # measure data loading time
            data_time.update(time.time() - end)

            inputs, targets = inputs.cuda(), targets.cuda()
            bs = inputs.size(0)

            # compute output
            _, mu, std, out_digit = model(inputs)
            cross_loss = criterion(out_digit, targets)
            info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean()
            loss = cross_loss + beta * info_loss

            # measure accuracy and record loss
            prec1, prec5 = accuracy(out_digit.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), bs)
            top1.update(prec1.item(), bs)
            top5.update(prec5.item(), bs)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            msg = '({batch}/{size}) | ' \
                  'Loss:{loss:.4f} | ' \
                  'top1:{top1: .4f} | top5:{top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                cls=losses.avg,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            pbar.set_description(msg)
    return losses.avg, top1.avg

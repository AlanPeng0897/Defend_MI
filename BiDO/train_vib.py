import torch, os, engine, model, utils, sys
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from util import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from copy import deepcopy

device = "cuda"


def main(args, trainloader, testloader):
    mode = "vib"
    root_path = "./"
    log_path = os.path.join(root_path, "target_logs")
    log_file = f"{mode}.txt"
    utils.Tee(os.path.join(log_path, log_file), 'a+')

    model_name = args["dataset"]["model_name"]
    n_epochs = args[model_name]["epochs"]
    weight_decay = args[model_name]["weight_decay"]
    momentum = args[model_name]["momentum"]
    lr = args[model_name]["lr"]
    milestones = args[model_name]["adjust_epochs"]
    n_classes = args["dataset"]["n_classes"]

    if model_name == "VGG16_vib":
        net = model.VGG16_vib(n_classes)

    elif model_name == "ResNet":
        # net = model.ResNetCls(nclass=n_classes, dropout=0)
        net = model.PretrainedResNet(nc=1, nclass=n_classes, imagesize=128)

    elif model_name == "MCNN":
        net = model.MCNN_vib(n_classes)

    optimizer = torch.optim.SGD(params=net.parameters(),
                                lr=lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    # scheduler = MultiStepLR(optimizer, milestones, gamma=0.1)

    criterion = nn.CrossEntropyLoss().cuda()
    net = torch.nn.DataParallel(net).to(device)

    best_ACC = -1

    for epoch in range(n_epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, n_epochs, lr))
        train_loss, train_acc = engine.train_vib(net, criterion, optimizer, trainloader, beta)
        test_loss, test_acc = engine.test_vib(net, criterion, testloader, beta)

        if test_acc > best_ACC:
            best_ACC = test_acc
            best_model = deepcopy(net)

        # scheduler.step()

    print("best acc:", best_ACC)
    utils.save_checkpoint({
        'state_dict': best_model.state_dict(),
    }, model_path, "{}_beta{:.3f}_{:.2f}.tar".format(model_name, beta, best_ACC))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='HSIC-COCO')
    parser.add_argument('--dataset', default='celeba', help='celeba | mnist')
    args = parser.parse_args()

    # dataset_name = "celeba"
    # dataset_name = "mnist"
    # dataset_name = "chestxray"
    # dataset_name = "facescrub"
    # dataset_name = "cifar"
    dataset_name = args.dataset

    root_path = "./"
    file = "./config/" + dataset_name + ".json"
    args = utils.load_json(json_file=file)

    model_name = args["dataset"]["model_name"]
    model_path = os.path.join(root_path, "target_model/" + dataset_name)
    os.makedirs(model_path, exist_ok=True)

    print("---------------------Training [%s]---------------------" % model_name)
    # utils.print_params(args["dataset"], args[model_name], dataset=args['dataset']['name'])

    # l = [0.001, 0.005, 0.01, 0.02, 0.05]
    # l = [0.015]
    args[model_name]["batch_size"] = 64
    # l = [0.16, 0.17, 0.18, 0.19, 0.21, 0.22, 0.23, 0.24, 0.25]
    l = [0.1]
    print('beta in ', l)
    for beta in l:
        print('beta =', beta)
        train_file = args['dataset']['train_file']
        test_file = args['dataset']['test_file']
        trainloader = utils.init_dataloader(args, train_file, mode="train")
        testloader = utils.init_dataloader(args, test_file, mode="test")

        main(args, trainloader, testloader)

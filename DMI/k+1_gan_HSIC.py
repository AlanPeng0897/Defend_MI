import os
import time
import utils
import torch
from utils import *
from torch.autograd import grad
import torch.nn.functional as F
from discri import DGWGAN, Discriminator, MinibatchDiscriminator
from discri import *
from generator import Generator
from generator import *
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import sys

sys.path.append('../BiDO')
import model


def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False)


def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)


def gradient_penalty(x, y):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = x + alpha * (y - x)
    z = z.cuda()
    z.requires_grad = True

    o = DG(z)
    g = grad(o, z, grad_outputs=torch.ones(o.size()).cuda(), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()

    return gp


def log_sum_exp(x, axis=1):
    m = torch.max(x, dim=1)[0]

    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim=axis))


if __name__ == "__main__":
    parser = ArgumentParser(description='Step2: targeted recovery')
    parser.add_argument('--dataset', default='celeba', help='celeba | mnist')
    parser.add_argument('--defense', default='HSIC', help='HSIC | vib')
    parser.add_argument('--root_path', default="./improvedGAN")
    parser.add_argument('--model_path', default='../BiDO/target_model')
    parser.add_argument('--beta', default=0, type=float)
    parser.add_argument('--acc', default=0, type=float)
    args = parser.parse_args()

    file = "./config/" + args.dataset + ".json"
    loaded_args = load_json(json_file=file)

    ############################# mkdirs ##############################
    os.makedirs(args.root_path, exist_ok=True)
    save_model_dir = os.path.join(args.root_path, args.dataset, args.defense)
    os.makedirs(save_model_dir, exist_ok=True)

    ############################# mkdirs ##############################

    file_path = loaded_args['dataset']['gan_file_path']
    stage = loaded_args['dataset']['stage']
    lr = loaded_args[stage]['lr']
    batch_size = loaded_args[stage]['batch_size']
    z_dim = loaded_args[stage]['z_dim']
    epochs = loaded_args[stage]['epochs']
    n_critic = loaded_args[stage]['n_critic']

    utils.print_params(loaded_args["dataset"], loaded_args[stage])
    model_name = loaded_args["dataset"]["model_name"]

    if args.dataset == 'celeba':
        hp_ac_list = [
            # celeba - hsic
            # (0.01, 0.05, 84.81),
            # (0.01, 0.2, 81.65),
            # (0.01, 0.5, 74.97),

            # (0.05, 0.25, 83.84),
            # (0.05, 0.5, 80.35),
            # (0.05, 1.0, 70.31),
            # (0.05, 2.5, 53.49),
        ]
        for (a1, a2, ac) in hp_ac_list:
            print("a1:", a1, "a2:", a2, "test_acc:", ac)

            T = model.VGG16(1000, True)
            T = torch.nn.DataParallel(T).cuda()
            path_T = os.path.join(args.model_path, f"{args.dataset}/",args.defense,
                                  "{}_{:.3f}&{:.3f}_{:.2f}.tar".format(model_name, a1, a2, ac))

            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'], strict=False)

            print("---------------------Training [%s]------------------------------" % stage)

            dataset, dataloader = utils.init_dataloader(loaded_args, file_path, batch_size, mode="gan")

            G = Generator(z_dim)
            DG = MinibatchDiscriminator()
            G = torch.nn.DataParallel(G).cuda()
            DG = torch.nn.DataParallel(DG).cuda()

            dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
            g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

            entropy = HLoss()

            step = 0

            for epoch in range(0, epochs):
                start = time.time()
                _, unlabel_loader1 = init_dataloader(loaded_args, file_path, batch_size, mode="gan", iterator=True)
                _, unlabel_loader2 = init_dataloader(loaded_args, file_path, batch_size, mode="gan", iterator=True)

                for i, imgs in enumerate(dataloader):
                    current_iter = epoch * len(dataloader) + i + 1

                    step += 1
                    imgs = imgs.cuda()
                    bs = imgs.size(0)
                    x_unlabel = unlabel_loader1.next()
                    x_unlabel2 = unlabel_loader2.next()

                    freeze(G)
                    unfreeze(DG)

                    z = torch.randn(bs, z_dim).cuda()
                    f_imgs = G(z)

                    y_prob = T(imgs)[-1]

                    targetprobs = nn.functional.softmax(y_prob, dim=1)
                    # print(entropy(targetprobs))

                    y = torch.argmax(y_prob, dim=1).view(-1)

                    _, output_label = DG(imgs)
                    _, output_unlabel = DG(x_unlabel)
                    _, output_fake = DG(f_imgs)

                    loss_lab = softXEnt(output_label, y_prob)
                    loss_unlab = 0.5 * (torch.mean(F.softplus(log_sum_exp(output_unlabel)))
                                        - torch.mean(log_sum_exp(output_unlabel))
                                        + torch.mean(F.softplus(log_sum_exp(output_fake))))
                    dg_loss = loss_lab + loss_unlab

                    acc = torch.mean((output_label.max(1)[1] == y).float())

                    dg_optimizer.zero_grad()
                    dg_loss.backward()
                    dg_optimizer.step()

                    # train G
                    if step % n_critic == 0:
                        freeze(DG)
                        unfreeze(G)
                        z = torch.randn(bs, z_dim).cuda()
                        f_imgs = G(z)
                        mom_gen, output_fake = DG(f_imgs)
                        mom_unlabel, _ = DG(x_unlabel2)

                        mom_gen = torch.mean(mom_gen, dim=0)
                        mom_unlabel = torch.mean(mom_unlabel, dim=0)

                        Hloss = entropy(output_fake)
                        g_loss = torch.mean((mom_gen - mom_unlabel).abs()) + 1e-4 * Hloss

                        g_optimizer.zero_grad()
                        g_loss.backward()
                        g_optimizer.step()

                end = time.time()
                interval = end - start

                print("Epoch:%d \tTime:%.2f\tD_loss:%.2f\tG_loss:%.2f\t train_acc:%.2f" % (
                    epoch, interval, dg_loss, g_loss,
                    acc))

                if epoch + 1 >= 100 and (epoch + 1) % 10 == 0:
                    Gpath = os.path.join(save_model_dir, "{}_G_{:.3f}&{:.3f}_{:.2f}.tar").format(model_name, a1, a2, ac)
                    Dpath = os.path.join(save_model_dir, "{}_D_{:.3f}&{:.3f}_{:.2f}.tar").format(model_name, a1, a2, ac)

                    torch.save({'state_dict': G.state_dict()}, Gpath)
                    torch.save({'state_dict': DG.state_dict()}, Dpath)

    elif args.dataset == 'mnist':
        hp_ac_list = [
            # # mnist-coco
            (1, 50, 99.51),
        ]
        for (a1, a2, ac) in hp_ac_list:
            print("a1:", a1, "a2:", a2, "test_acc:", ac)
            T = model.MCNN(5)
            T = torch.nn.DataParallel(T).cuda()
            path_T = os.path.join(args.model_path, f"{args.dataset}",args.defense,
                                  "{}_{:.3f}&{:.3f}_{:.2f}.tar".format(model_name, a1, a2, ac))
            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'], strict=False)
            dataset, dataloader = utils.init_dataloader(loaded_args, file_path, batch_size, mode="gan")

            G = GeneratorMNIST(z_dim)
            G = torch.nn.DataParallel(G).cuda()
            DG = MinibatchDiscriminator_MNIST()
            DG = torch.nn.DataParallel(DG).cuda()

            dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
            g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

            entropy = HLoss()

            step = 0
            for epoch in range(0, epochs):
                start = time.time()
                _, unlabel_loader1 = init_dataloader(loaded_args, file_path, batch_size, mode="gan", iterator=True)
                _, unlabel_loader2 = init_dataloader(loaded_args, file_path, batch_size, mode="gan", iterator=True)

                for i, imgs in enumerate(dataloader):
                    current_iter = epoch * len(dataloader) + i + 1

                    step += 1
                    imgs = imgs.cuda()
                    bs = imgs.size(0)
                    x_unlabel = unlabel_loader1.next()
                    x_unlabel2 = unlabel_loader2.next()

                    freeze(G)
                    unfreeze(DG)

                    z = torch.randn(bs, z_dim).cuda()
                    f_imgs = G(z)

                    y_prob = T(imgs)[-1]

                    targetprobs = nn.functional.softmax(y_prob, dim=1)
                    # print(entropy(targetprobs))

                    y = torch.argmax(y_prob, dim=1).view(-1)

                    _, output_label = DG(imgs)
                    _, output_unlabel = DG(x_unlabel)
                    _, output_fake = DG(f_imgs)

                    loss_lab = softXEnt(output_label, y_prob)
                    loss_unlab = 0.5 * (torch.mean(F.softplus(log_sum_exp(output_unlabel)))
                                        - torch.mean(log_sum_exp(output_unlabel))
                                        + torch.mean(F.softplus(log_sum_exp(output_fake))))
                    dg_loss = loss_lab + loss_unlab

                    acc = torch.mean((output_label.max(1)[1] == y).float())

                    dg_optimizer.zero_grad()
                    dg_loss.backward()
                    dg_optimizer.step()

                    # train G
                    if step % n_critic == 0:
                        freeze(DG)
                        unfreeze(G)
                        z = torch.randn(bs, z_dim).cuda()
                        f_imgs = G(z)
                        mom_gen, output_fake = DG(f_imgs)
                        mom_unlabel, _ = DG(x_unlabel2)

                        mom_gen = torch.mean(mom_gen, dim=0)
                        mom_unlabel = torch.mean(mom_unlabel, dim=0)

                        Hloss = entropy(output_fake)
                        g_loss = torch.mean((mom_gen - mom_unlabel).abs()) + 1e-4 * Hloss

                        g_optimizer.zero_grad()
                        g_loss.backward()
                        g_optimizer.step()

                end = time.time()
                interval = end - start

                print("Epoch:%d \tTime:%.2f\tD_loss:%.2f\tG_loss:%.2f\t train_acc:%.2f" % (
                    epoch, interval, dg_loss, g_loss,
                    acc))

                if epoch + 1 >= 100 and (epoch + 1) % 10 == 0:
                    Gpath = os.path.join(save_model_dir, "{}_G_{:.3f}&{:.3f}_{:.2f}.tar").format(model_name, a1, a2, ac)
                    Dpath = os.path.join(save_model_dir, "{}_D_{:.3f}&{:.3f}_{:.2f}.tar").format(model_name, a1, a2, ac)

                    torch.save({'state_dict': G.state_dict()}, Gpath)
                    torch.save({'state_dict': DG.state_dict()}, Dpath)





import os, sys
import time
import utils
import torch
import dataloader
import torchvision
from utils import *
from torch.autograd import grad
import torch.nn.functional as F
from discri import *
from generator import *
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

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
    parser = ArgumentParser(description='Step1: targeted recovery')
    parser.add_argument('--dataset', default='celeba', help='celeba | mnist')
    parser.add_argument('--defense', default='reg', help='reg | vib')
    parser.add_argument('--root_path', default="./improvedGAN")
    parser.add_argument('--model_path', default='../BiDO/target_model')
    parser.add_argument('--beta', default=0, type=float)
    parser.add_argument('--acc', default=0, type=float)
    args = parser.parse_args()

    file = "./config/" + args.dataset + ".json"
    loaded_args = load_json(json_file=file)
    ############################# mkdirs ##############################
    save_model_dir = os.path.join(args.root_path, args.dataset, args.defense)
    os.makedirs(save_model_dir, exist_ok=True)
    save_img_dir = "./improvedGAN/imgs_improved_{}".format(args.dataset)
    os.makedirs(save_img_dir, exist_ok=True)
    ############################# mkdirs ##############################

    file_path = loaded_args['dataset']['gan_file_path']
    stage = loaded_args['dataset']['stage']
    lr = loaded_args[stage]['lr']
    batch_size = loaded_args[stage]['batch_size']
    z_dim = loaded_args[stage]['z_dim']
    epochs = loaded_args[stage]['epochs']
    n_critic = loaded_args[stage]['n_critic']
    n_classes = loaded_args["dataset"]["n_classes"]

    model_name = loaded_args["dataset"]["model_name"]

    if args.dataset == 'celeba':
        if args.defense == 'vib':
            # beta, ac = 1e-3, 83.42
            # beta, ac = 3e-3, 78.70
            beta = args.beta
            ac = args.acc
            T = model.VGG16_vib(n_classes)
            T = torch.nn.DataParallel(T).cuda()
            path_T = os.path.join("../BiDO/target_model/{}".format(args.dataset), args.defense,
                                  f"{model_name}_beta{beta:.3f}_{ac:.2f}.tar")

            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'])

            Gpath = os.path.join(save_model_dir, "{}_G_beta_{:.3f}_{:.2f}.tar").format(model_name, beta, ac)
            Dpath = os.path.join(save_model_dir, "{}_D_beta_{:.3f}_{:.2f}.tar").format(model_name, beta, ac)

        elif args.defense == 'reg':
            ac = 86.21
            T = model.VGG16(n_classes)
            T = torch.nn.DataParallel(T).cuda()
            path_T = os.path.join(args.model_path, args.dataset, args.defense, f"VGG16_reg_{ac:.2f}.tar")

            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'])

            Gpath = os.path.join(save_model_dir, "{}_G_reg_{:.2f}.tar").format(model_name, ac)
            Dpath = os.path.join(save_model_dir, "{}_D_reg_{:.2f}.tar").format(model_name, ac)

    elif args.dataset == 'mnist':
        if args.defense == 'vib':
            beta = args.beta
            ac = args.acc
            T = model.MCNN_vib(n_classes)
            T = torch.nn.DataParallel(T).cuda()
            path_T = os.path.join("../BiDO/target_model/{}".format(args.dataset),args.defense,
                                  f"{model_name}_beta{beta:.3f}_{ac:.2f}.tar")

            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'])

            Gpath = os.path.join(save_model_dir, "{}_G_beta_{:.3f}_{:.2f}.tar").format(model_name, beta, ac)
            Dpath = os.path.join(save_model_dir, "{}_D_beta_{:.3f}_{:.2f}.tar").format(model_name, beta, ac)

        if args.defense == 'reg':
            ac = 99.94
            T = model.MCNN(n_classes)
            T = torch.nn.DataParallel(T).cuda()
            path_T = os.path.join(args.model_path, args.dataset, args.defense, "MCNN_reg_99.94.tar")
            ckp_T = torch.load(path_T)
            check = T.load_state_dict(ckp_T['state_dict'])

            Gpath = os.path.join(save_model_dir, "{}_G_reg_{:.2f}.tar").format(model_name, ac)
            Dpath = os.path.join(save_model_dir, "{}_D_reg_{:.2f}.tar").format(model_name, ac)

    elif args.dataset == 'cifar':
        if args.defense == 'reg':
            ac = args.acc
            T = VGG16(n_classes, args.dataset)
            T = torch.nn.DataParallel(T).cuda()
            path_T = os.path.join(args.model_path, args.dataset, f"VGG16_reg_{ac:.2f}.tar")

            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'])

            Gpath = os.path.join(save_model_dir, "{}_G_reg_{:.2f}.tar").format(model_name, ac)
            Dpath = os.path.join(save_model_dir, "{}_D_reg_{:.2f}.tar").format(model_name, ac)

        else:
            beta = args.beta
            ac = args.acc
            T = VGG16_vib(n_classes, args.dataset)
            T = torch.nn.DataParallel(T).cuda()
            path_T = os.path.join("../BiDO/target_model/{}".format(args.dataset),
                                  f"{model_name}_beta{beta:.3f}_{ac:.2f}.tar")
            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'])
            Gpath = os.path.join(save_model_dir, "{}_G_beta_{:.3f}_{:.2f}.tar").format(model_name, beta, ac)
            Dpath = os.path.join(save_model_dir, "{}_D_beta_{:.3f}_{:.2f}.tar").format(model_name, beta, ac)

    elif args.dataset == 'chestxray':
        ac = 45.90
        T = model.ResNetCls(nclass=n_classes)
        path_T = os.path.join(args.model_path, args.dataset, "best_ckpt.pt")
        ckp_T = torch.load(path_T)
        T.load_state_dict(ckp_T, strict=False)
        T = torch.nn.DataParallel(T).cuda()

        Gpath = os.path.join(save_model_dir, "{}_G_reg_{:.2f}.tar").format(model_name, ac)
        Dpath = os.path.join(save_model_dir, "{}_D_reg_{:.2f}.tar").format(model_name, ac)

    print("---------------------Training [%s]------------------------------" % stage)

    if args.dataset == 'celeba':
        G = Generator(z_dim)
        DG = MinibatchDiscriminator()

    elif args.dataset == 'mnist':
        G = GeneratorMNIST(z_dim)
        DG = MinibatchDiscriminator_MNIST()

    elif args.dataset == 'chestxray':
        G = GeneratorChestX(z_dim)
        DG = MinibatchDiscriminator_ChestX()

    elif args.dataset == 'cifar':
        G = GeneratorCIFAR(z_dim)
        DG = MinibatchDiscriminator_CIFAR()

    G = torch.nn.DataParallel(G).cuda()
    DG = torch.nn.DataParallel(DG).cuda()
    dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    entropy = HLoss()

    if args.dataset == 'chestxray':
        sys.path.append('../VMI')
        import chestxray

        bs = 64
        imgSize = 128
        aux_x = chestxray.load_aux_cache(imgSize)
        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(aux_x),
                                                 batch_size=bs, shuffle=True, num_workers=2)
    else:
        _, dataloader = utils.init_dataloader(loaded_args, file_path, batch_size, mode="gan")

    step = 0
    for epoch in range(0, epochs):
        start = time.time()

        if args.dataset == 'chestxray':
            unlabel_loader1 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(aux_x),
                                                          batch_size=bs, shuffle=True, num_workers=2).__iter__()
            unlabel_loader2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(aux_x),
                                                          batch_size=bs, shuffle=True, num_workers=2).__iter__()
        else:
            _, unlabel_loader1 = init_dataloader(loaded_args, file_path, batch_size, mode="gan", iterator=True)
            _, unlabel_loader2 = init_dataloader(loaded_args, file_path, batch_size, mode="gan", iterator=True)

        for i, imgs in enumerate(dataloader):
            if isinstance(imgs, list):
                imgs = imgs[0]
            current_iter = epoch * len(dataloader) + i + 1
            step += 1
            imgs = imgs.cuda()
            bs = imgs.size(0)
            x_unlabel = unlabel_loader1.next()
            x_unlabel2 = unlabel_loader2.next()

            if isinstance(x_unlabel, list):
                x_unlabel = x_unlabel[0]
                x_unlabel2 = x_unlabel2[0]

            freeze(G)
            unfreeze(DG)

            z = torch.randn(bs, z_dim).cuda()
            f_imgs = G(z)

            y_prob = T(imgs)[-1]

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

        print("Epoch:%d \tTime:%.2f\tD_loss:%.2f\tG_loss:%.2f\t train_acc:%.2f" % (epoch, interval, dg_loss, g_loss,
                                                                                   acc))

        # if epoch + 1 >= 100 and (epoch + 1) % 10 == 0:
        torch.save({'state_dict': G.state_dict()}, Gpath)
        torch.save({'state_dict': DG.state_dict()}, Dpath)

        if (epoch + 1) % 5 == 0:
            z = torch.randn(32, z_dim).cuda()
            fake_image = G(z)
            save_tensor_images(fake_image.detach(),
                               os.path.join(save_img_dir,
                                            f"improved_{args.dataset}_img_{args.defense}_{epoch + 1}.png"),
                               nrow=8)

import os
import time
import utils
import torch
from utils import save_tensor_images, init_dataloader, load_json
from torch.autograd import grad
from discri import DGWGAN, DGWGAN32
from generator import Generator, GeneratorMNIST, GeneratorCXR
from argparse import ArgumentParser


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


if __name__ == "__main__":
    parser = ArgumentParser(description='Step1: train GAN')
    parser.add_argument('--dataset', default='celeba', help='celeba | mnist')

    args = parser.parse_args()

    ############################# mkdirs ##############################
    save_model_dir = f"result/models_{args.dataset}_gan"
    os.makedirs(save_model_dir, exist_ok=True)
    save_img_dir = f"result/imgs_{args.dataset}_gan"
    os.makedirs(save_img_dir, exist_ok=True)
    ############################# mkdirs ##############################

    file = "./config/" + args.dataset + ".json"
    loaded_args = load_json(json_file=file)
    file_path = loaded_args['dataset']['train_file_path']
    model_name = loaded_args['dataset']['model_name']
    lr = loaded_args[model_name]['lr']
    batch_size = loaded_args[model_name]['batch_size']
    z_dim = loaded_args[model_name]['z_dim']
    epochs = loaded_args[model_name]['epochs']
    n_critic = loaded_args[model_name]['n_critic']

    print("---------------------Training [%s]------------------------------" % model_name)
    utils.print_params(loaded_args["dataset"], loaded_args[model_name])

    dataset, dataloader = init_dataloader(loaded_args, file_path, batch_size, mode="gan")

    if args.dataset == 'celeba':
        G = Generator(z_dim)
        DG = DGWGAN(3)
    elif args.dataset == 'mnist':
        G = GeneratorMNIST(z_dim)
        DG = DGWGAN32()

    G = torch.nn.DataParallel(G).cuda()
    DG = torch.nn.DataParallel(DG).cuda()

    # 0.004
    dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    step = 0
    for epoch in range(epochs):
        start = time.time()
        for i, imgs in enumerate(dataloader):
            step += 1
            imgs = imgs.cuda()
            bs = imgs.size(0)

            freeze(G)
            unfreeze(DG)

            z = torch.randn(bs, z_dim).cuda()
            f_imgs = G(z)

            r_logit = DG(imgs)
            f_logit = DG(f_imgs)

            wd = r_logit.mean() - f_logit.mean()  # Wasserstein-1 Distance
            gp = gradient_penalty(imgs.data, f_imgs.data)
            dg_loss = - wd + gp * 10.0

            dg_optimizer.zero_grad()
            dg_loss.backward()
            dg_optimizer.step()

            # train G
            if step % n_critic == 0:
                freeze(DG)
                unfreeze(G)
                z = torch.randn(bs, z_dim).cuda()
                f_imgs = G(z)
                logit_dg = DG(f_imgs)
                # calculate g_loss
                g_loss = - logit_dg.mean()

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

        end = time.time()
        interval = end - start
        print("Epoch:%d \tTime:%.2f\tD_loss:%.2f\tG_loss:%.2f" % (epoch, interval, dg_loss, g_loss))
        if (epoch + 1) % 10 == 0:
            z = torch.randn(32, z_dim).cuda()
            fake_image = G(z)
            save_tensor_images(fake_image.detach(), os.path.join(save_img_dir, "result_image_{}.png".format(epoch + 1)),
                               nrow=8)
        if epoch + 1 >= 100:
            print('saving weights file')
            torch.save({'state_dict': G.state_dict()},
                       os.path.join(save_model_dir, f"{args.dataset}_G.tar"))
            torch.save({'state_dict': DG.state_dict()},
                       os.path.join(save_model_dir, f"{args.dataset}_D.tar"))

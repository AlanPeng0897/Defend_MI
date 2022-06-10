import os
import re
from typing import List

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

# Attack
import sys

sys.path.append('../boosted-implicit-models')
from experimental import AttackExperiment
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pylab as  plt
from tqdm import tqdm


# ----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(',')
    return [int(x) for x in vals]


# ----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--styles', 'col_styles', type=num_range, help='Style layer range', default='0-6', show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const',
              show_default=True)
@click.option('--outdir', type=str, required=True)
@click.option('--fixed_id', type=int, required=True)
@click.option('--noise', type=bool)
def attack(
        network_pkl: str,
        col_styles: List[int],
        truncation_psi: float,
        noise_mode: str,
        outdir: str,
        fixed_id: int,
        noise: bool
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    python attack.py --outdir=out-celeba --network=training-runs/00015-celeba-aux-auto1/network-snapshot-000240.pkl --trunc .5
    """
    device = torch.device('cuda')

    # Prepare attack settings
    # experiment = AttackExperiment('/h/wangkuan/projects/boosted-implicit-models/configs/celeba_crop--dcgan--ResNet10--ft.yml', device, False, fixed_id =0, run_target_feat_eval=0)
    experiment = AttackExperiment(
        '/h/wangkuan/projects/boosted-implicit-models/configs/celeba_db--dcgan--ResNet10--ft.yml', device, False,
        fixed_id=fixed_id, run_target_feat_eval=0)
    target_logsoftmax = experiment.target_logsoftmax

    #
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, 'samples_pt'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'ws_pt'), exist_ok=True)

    print('Generating W vectors...')

    """
    # with 1 z for each layer

    zs = torch.randn(N, L, Z)
    ws = G.mapping(z.reshape(-1, zdim), None)  # (N * L, L, zdim)
    w = w[:, 0].reshape(N, -1, zdim) # (N, L, Z)

    G.synthesis(w)
    """
    if noise:
        all_z = torch.randn((100, G.z_dim), device=device, requires_grad=True)
        optimizerG = optim.Adam([all_z], lr=0.01)

    else:
        with torch.no_grad():
            all_z = np.random.randn(100, G.z_dim)
            all_w = G.mapping(torch.from_numpy(all_z).to(device), None)
            w_avg = G.mapping.w_avg
            all_w = w_avg + (all_w - w_avg) * truncation_psi
        style_mask = torch.zeros(1, all_w.shape[1], 1).to(device)
        style_mask[:, col_styles, :] = 1
        opt_w = all_w.clone()
        opt_w.requires_grad_()
        optimizerG = optim.Adam([opt_w], lr=0.01)

        # print('Generating images...')

    # all_images = G.synthesis(all_w, noise_mode=noise_mode)

    # all_images = (all_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    def save_images(all_images, fpath):
        fakegrid = vutils.make_grid(all_images[:100].clamp(-1, 1) * .5 + .5, nrow=10, padding=4, pad_value=1,
                                    normalize=False)
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(np.transpose(fakegrid.cpu().numpy(), (1, 2, 0)), interpolation='bilinear')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(fpath, bbox_inches='tight', pad_inches=0)

    pbar = tqdm(range(0, 10000), desc='Train loop')
    for i in pbar:
        optimizerG.zero_grad()
        # all_w[:, col_styles] = opt_w
        if noise:
            w = G.mapping(all_z)
        else:
            w = (1 - style_mask) * all_w + style_mask * opt_w
        fake = G.synthesis(w, noise_mode=noise_mode)
        # import ipdb; ipdb.set_trace()
        lsm = target_logsoftmax(fake.clamp(-1, 1) / 2 + .5)
        fake_y = fixed_id * torch.ones(100).to(device).long()
        target_loss = -lsm.gather(1, fake_y.view(-1, 1)).mean()
        target_loss.backward()
        optimizerG.step()
        if i % 1000 == 0:
            save_images(fake, f'{outdir}/i{i:05d}.jpeg')
            torch.save(fake[:10].detach().cpu(), os.path.join(outdir, 'samples_pt', f'i{i:05d}.pt'))
            torch.save(opt_w.detach().cpu(), os.path.join(outdir, 'ws_pt', f'i{i:05d}.pt'))


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    attack()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------

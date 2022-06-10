import os
import re
from typing import List

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import legacy

# Attack
import sys
sys.path.append('../boosted-implicit-models')
from experimental import AttackExperiment
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pylab as  plt
from tqdm import tqdm
from likelihood_model import ReparameterizedMVN


#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

def save_images(all_images, fpath):
        fakegrid = vutils.make_grid(all_images[:100].clamp(-1,1) * .5 + .5, nrow=10, padding=4, pad_value=1, normalize=False) 
        fig, ax = plt.subplots(1,1,figsize=(12,12))
        ax.imshow(np.transpose(fakegrid.cpu().numpy(), (1,2,0)), interpolation='bilinear')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(fpath, bbox_inches='tight', pad_inches=0)


class MineGAN(nn.Module):
    def __init__(self, miner, Gmapping):
        super(MineGAN, self).__init__()
        self.nz = miner.nz0
        self.miner = miner
        self.Gmapping = Gmapping

    def forward(self, z0):
        z = self.miner(z0) 
        w = self.Gmapping(z, None) 
        return w


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--layers', 'l_identity', type=num_range, help='Style layer range', default='0-6', show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', type=str, required=True)
@click.option('--fixed_id', type=int, required=True)
def attack(
    network_pkl: str,
    l_identity: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    fixed_id: int
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    python attack.py --outdir=out-celeba --network=training-runs/00015-celeba-aux-auto1/network-snapshot-000240.pkl --trunc .5
    """
    device = torch.device('cuda')

    # Prepare attack settings
    # experiment = AttackExperiment('/h/wangkuan/projects/boosted-implicit-models/configs/celeba_crop--dcgan--ResNet10--ft.yml', device, False, fixed_id =0, run_target_feat_eval=0)
    experiment = AttackExperiment('/h/wangkuan/projects/boosted-implicit-models/configs/celeba_db--dcgan--ResNet10--ft.yml', device, False, fixed_id =fixed_id, run_target_feat_eval=0)
    target_logsoftmax = experiment.target_logsoftmax

    assert truncation_psi == 1 # TODO: incorporate this if necessary

    #
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, 'samples_pt'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'ws_pt'), exist_ok=True)

    miner = ReparameterizedMVN(G.mapping.z_dim).to(device).double()

    minegan_Gmapping = MineGAN(miner, G.mapping)

    identity_mask = torch.zeros(1, G.mapping.num_ws, 1).to(device)
    identity_mask[:, l_identity, :] = 1

    optimizerG = optim.Adam(miner.parameters(), lr=0.01) 

    fixed_z_nuisance = torch.randn(100, G.z_dim).to(device).double()
    fixed_z_identity = torch.randn(100, G.z_dim).to(device).double()
    
    pbar = tqdm(range(0, 10000), desc='Train loop')
    for i in pbar:
        optimizerG.zero_grad()

        def sample(z_nuisance, z_identity):
            w_nuisance = G.mapping(z_nuisance, None)
            w_identity = minegan_Gmapping(z_identity)
            w = (1-identity_mask) * w_nuisance + identity_mask * w_identity
            x = G.synthesis(w, noise_mode=noise_mode)
            return x
        # all_w[:, l_identity] = opt_w
        z_nu = torch.randn(100, G.z_dim).to(device).double()
        z_id = torch.randn(100, G.z_dim).to(device).double()
        fake = sample(z_nu, z_id)
        # import ipdb; ipdb.set_trace()
        lsm =  target_logsoftmax(fake.clamp(-1,1) / 2 + .5)
        fake_y =  fixed_id * torch.ones(100).to(device).long()
        target_loss = -lsm.gather(1, fake_y.view(-1,1)).mean()
        target_loss.backward()
        optimizerG.step()
        if i % 100 == 0:
            with torch.no_grad():
                fake = sample(fixed_z_nuisance, fixed_z_identity)
            save_images(fake, f'{outdir}/i{i:05d}.jpeg')
            torch.save( fake[:10].detach().cpu(), os.path.join(outdir, 'samples_pt', f'i{i:05d}.pt'))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    attack() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------

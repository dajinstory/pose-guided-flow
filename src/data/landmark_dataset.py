import os, sys
import numpy as np
import pandas as pd
import math, random
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

# from util import computeGaussian, draw_edge
from .base_dataset import BaseDataset

tt = T.ToTensor()
# ttp = T.ToPILImage()
# ptt = T.PILToTensor()

class LandmarkDataset(BaseDataset):
    def __init__(self, opt):
        self.root = opt['root_path']
        self.frames = self._prepare_frames(opt)
        self.img_size = opt['in_size']
        self.norm = False
        self.noise = False
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # Verify data
        idx = self._verify_data(idx, range(len(self.frames)))
        
        im = self.load_image(idx)
        ldmk = self.load_landmark(idx)
            
        return im, ldmk

    def load_image(self, idx):
       # Load images
        im = self.frames[idx]['name']
        im = Image.open(os.path.join(self.root_path, im))

        # Resize image if necessary
        if im.size[0] != self.img_size and im.size[1] != self.img_size:
            im = im.resize((self.img_size, self.img_size))
        im = self.to_tensor(im)

        # Grayscale Exception
        if im.shape[0] != 3:
            im = torch.stack([im[0,:,:]]*3, dim=0)

        return im

    def load_landmark(self, idx):
        # Load landmarks
        ldmk_path = self.frames[idx]['name'].replace('.png', 'ldmk.npy')
        f5p_path = self.frames[idx]['name'].replace('.png', 'f5p.npy')
        ldmk = np.load(os.path.join(self.ldmk_root_path, ldmk_path))
        f5p = np.load(os.path.join(self.ldmk_root_path, f5p_path))

        ldmk = torch.Tensor(ldmk)
        f5p = torch.Tensor(f5p)

        # Case #3 : 68-channel + 1-channel
        conditions = []
        res = self.img_size
        for _ in range(7): 
            heatmap = self._computeGaussian(p=ldmk, res=res, kernel_sigma=0.1)
            edgemap = self._draw_edge(ldmk, img_size=res)
            conditions.append(torch.cat([heatmap, edgemap], dim=0))
            res = res // 2

        return conditions, ldmk, f5p

    def _computeGaussian(self, p, res=64, kernel_sigma=0.05, device='cpu'):
        '''
        REFERENCE : 못찾음...
        p should be torch.Tensor(), with shape (N_points, 2)
        each point should (0.,0.) ~ (1., 1.)
        '''
        ksize = round(res * kernel_sigma * 6)
        sigma = kernel_sigma * res
        x = np.linspace(0, 1, int(res))
        y = np.linspace(0, 1, int(res))
        xv, yv = np.meshgrid(x, y)
        txv = torch.from_numpy(xv).unsqueeze(0).float()
        tyv = torch.from_numpy(yv).unsqueeze(0).float()
        mesh = torch.cat((txv, tyv), 0).to(device)          # positions
        heatmap = torch.zeros((len(p), res, res)).to(device)    # create an empty gaussian density image

        for i in range(len(p)):
            center = p[i, 0:2]  # go through each point
            hw = round(3 * kernel_sigma * res)  # only consider band-limited gaussian with 3sigma
            coord_center = torch.floor(center * res)

            up = torch.max(coord_center[1] - hw, torch.Tensor([0]).to(device))  # take the boundary into account
            down = torch.min(coord_center[1] + hw + 1, torch.Tensor([res]).to(device))
            left = torch.max(coord_center[0] - hw, torch.Tensor([0]).to(device))
            right = torch.min(coord_center[0] + hw + 1, torch.Tensor([res]).to(device))
            up = up.long()
            down = down.long()
            left = left.long()
            right = right.long()

            # apply gaussian kernel on the pixels based on their distance to the center
            heatmap[i, up:down, left:right] = torch.exp(
                -(mesh.permute(1, 2, 0)[up:down, left:right, :] - center).pow(2).sum(2) / (2 * kernel_sigma ** 2))
        return heatmap
        
    def _draw_edge(self, ldmk, img_size=None):
        img_size = img_size if img_size is not None else self.img_size 
        n_partials = [17, 5, 5, 4, 5, 6, 6, 12, 8] # uface, lbrow, rbrow, hnose, wnose, leye, reye, mouth_out, mouth_in
        img  = Image.new( mode = "L", size = (img_size, img_size) )
        draw = ImageDraw.Draw(img)

        idx=0
        for n_partial in n_partials:
            x_s, y_s = torch.floor(ldmk[idx] * self.img_size)
            for x_e, y_e in ldmk[idx+1:idx+n_partial]:
                x_e = torch.floor(x_e * self.img_size)
                y_e = torch.floor(y_e * self.img_size)
                draw.line((x_s, y_s, x_e, y_e), fill=255)
                x_s, y_s = x_e, y_e
            idx += n_partial
        
        return tt(img)
    
    def _verify_data(self, idx, random_range):
        # Avoid landmark-missed data
        while math.isnan(self.frames[idx]['landmark_0_x']):
            idx = random.choice(random_range)
        return idx
import os, sys
import numpy as np
import pandas as pd
import math, random
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import ToPILImage, PILToTensor

from util import computeGaussian
from .base_dataset import BaseDataset

tt = T.ToTensor()
tti = ToPILImage()
itt = PILToTensor()

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

    def load_landmark(self, idx):
        # Load landmarks
        ldm = []
        for li in range(68):
            ldm_x = self.frames[idx]['landmark_%d_x'%(li)]
            ldm_y = self.frames[idx]['landmark_%d_y'%(li)]
            ldm.append([ldm_x, ldm_y])    
        ldm = np.array(ldm)
        ldm = torch.Tensor(ldm)

        # Case #1 : 1-channel Heatmap
        # heatmap = self._draw_edge(ldm)
       
        # Case #2 : 68-channel Heatmaps
        heatmaps = []
        res = self.img_size
        for _ in range(7): 
            heatmap = computeGaussian(ldm, res=res, kernel_sigma=0.1)
            heatmaps.append(heatmap)
            res = res // 2

        return heatmaps
        
    def _verify_data(self, idx, random_range):
        # Avoid landmark-missed data
        while math.isnan(self.frames[idx]['landmark_0_x']):
            idx = random.choice(random_range)
        return idx

    def _draw_edge(self, ldmk):
        n_partials = [17, 5, 5, 4, 5, 6, 6, 12, 8] # uface, lbrow, rbrow, hnose, wnose, leye, reye, mouth_out, mouth_in
        img  = Image.new( mode = "L", size = (self.img_size, self.img_size) )
        draw = ImageDraw.Draw(img)

        idx=0
        for n_partial in n_partials:
            x_s, y_s = (ldmk[idx] * self.img_size) // 1
            for x_e, y_e in ldmk[idx+1:idx+n_partial]:
                x_e = (x_e * self.img_size) // 1
                y_e = (y_e * self.img_size) // 1
                draw.line((x_s, y_s, x_e, y_e), fill=255)
                x_s, y_s = x_e, y_e
            idx += n_partial
        
        return tt(img)
    
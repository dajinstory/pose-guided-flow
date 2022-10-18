import os, sys
import numpy as np
import pandas as pd
import math, random
from PIL import Image, ImageDraw

import torch
from torchvision import transforms as T

from util import computeGaussian, draw_edge
from .landmark_dataset import LandmarkDataset

tt = T.ToTensor()
# ttp = ToPILImage()
# ptt = PILToTensor()

class TripletLandmarkDataset(LandmarkDataset):
    def __init__(self, opt):
        self.root_path = opt['root_path']
        self.ldmk_root_path = opt['ldmk_root_path']
        self.frames = self._prepare_frames(opt)
        self.img_size = opt['in_size']
        self.norm = False
        self.noise = False
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # Verify data
        # idx = self._verify_data(idx, range(len(self.frames)))
        
        # Select main frame, positive frame and negative frame
        l = self.frames[idx]['range_start']
        r = self.frames[idx]['range_end']
        p_idx = random.choice([*range(l, idx), *range(idx+1, r)])
        # p_idx = self._verify_data(p_idx, [*range(l, idx), *range(idx+1, r)])
        n_idx = random.choice([*range(0,l), *range(r, len(self.frames))])
        # n_idx = self._verify_data(n_idx, [*range(0,l), *range(r, len(self.frames))])
        
        # Prepare Dataset
        im0 = self.load_image(idx)
        im1 = self.load_image(p_idx)
        im2 = self.load_image(n_idx)
        c0, ldmk0, f5p0 = self.load_landmark(idx)
        c1, ldmk1, f5p1 = self.load_landmark(p_idx)
        c2, ldmk2, f5p2 = self.load_landmark(n_idx)

        return (im0, im1, im2), (c0, c1, c2), (ldmk0, ldmk1, ldmk2), (f5p0, f5p1, f5p2)

    def _create_small_meta(self, meta, n_batch):
        n_valid = n_batch * torch.cuda.device_count()
        stride = len(meta) // (n_valid//2)

        small_meta = []
        for idx in range(n_valid//2):
            # anchor
            a_idx = idx*stride
            anchor = meta[a_idx]

            # l, r
            l = anchor['range_start']
            r = anchor['range_end']
            p_idx = random.choice([*range(l, a_idx), *range(a_idx+1, r)])

            # pos
            pos = meta[p_idx]

            # update
            anchor['range_start'] = idx*2
            anchor['range_end'] = idx*2 + 2
            pos['range_start'] = idx*2
            pos['range_end'] = idx*2 + 2 

            small_meta.append(anchor)
            small_meta.append(pos)

        return small_meta
    
    
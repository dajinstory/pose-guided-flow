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
from .landmark_dataset import LandmarkDataset

tt = T.ToTensor()
tti = ToPILImage()
itt = PILToTensor()

class TripletLandmarkDataset(LandmarkDataset):
    def __init__(self, opt):
        super().__init__(opt)
    
    def __getitem__(self, idx):
        # Verify data
        idx = self._verify_data(idx, range(len(self.frames)))
        
        # Select main frame, positive frame and negative frame
        l = self.frames[idx]['range_start']
        r = self.frames[idx]['range_end']
        p_idx = random.choice([*range(l, idx), *range(idx+1, r)])
        p_idx = self._verify_data(p_idx, [*range(l, idx), *range(idx+1, r)])
        n_idx = random.choice([*range(0,l), *range(r, len(self.frames))])
        n_idx = self._verify_data(n_idx, [*range(0,l), *range(r, len(self.frames))])
        
        # Prepare Datset
        im0 = self.load_image(idx)
        im1 = self.load_image(p_idx)
        im2 = self.load_image(n_idx)
        ldm0 = self.load_landmark(idx)
        ldm1 = self.load_landmark(p_idx)
        ldm2 = self.load_landmark(n_idx)

        return (im0, im1, im2), (ldm0, ldm1, ldm2)
    
    
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



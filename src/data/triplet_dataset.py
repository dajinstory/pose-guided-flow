import os, sys
import numpy as np
import pandas as pd
import math, random
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import ToPILImage, PILToTensor

from util import computeGaussian, draw_edge

from .base_dataset import BaseDataset

tt = T.ToTensor()
tti = ToPILImage()
itt = PILToTensor()

class TripletDataset(Dataset):
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
        ldmk_path = self.frames[idx]['name'].replace('.jpg', 'ldmk.npy')
        f5p_path = self.frames[idx]['name'].replace('.jpg', 'f5p.npy')
        ldmk = np.load(os.path.join(self.ldmk_root_path, ldmk_path))
        f5p = np.load(os.path.join(self.ldmk_root_path, f5p_path))

        ldmk = torch.Tensor(ldmk)
        f5p = torch.Tensor(f5p)

        # Case #3 : 68-channel + 1-channel
        conditions = []
        res = self.img_size
        for _ in range(7): 
            heatmap = computeGaussian(ldmk, res=res, kernel_sigma=0.1)
            edgemap = draw_edge(ldmk, img_size=res)
            conditions.append(torch.cat([heatmap, edgemap], dim=0))
            res = res // 2

        return conditions, ldmk, f5p
    
    def _prepare_frames(self, opt):
        meta = pd.read_csv(opt['meta_path'], index_col=0).to_dict(orient='records')
        if opt['abbreviation']:
            meta = self._create_small_meta(meta, opt['batch_size_per_gpu'])
        return meta

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
    
    
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from .models.mobilefacenet import MobileFaceNet

ptt = T.ToTensor()
ttp = T.ToPILImage()

class FacialLandmarkDetector(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        
        # data preprocess
        self.img_size = 112
        self.scale = self.img_size / 112.

        # self.face_detector = Retinaface.Retinaface()            
        self.ldmk_detector = self._load_model(pretrained['ckpt_path'])
        self.ldmk_detector = self.ldmk_detector.eval()
        self.out_size = 112

    def _load_model(self, ckpt_path = 'checkpoint/mobilefacenet_model_best.pth.tar'):
        if torch.cuda.is_available():
            map_location='cuda'
        else:
            map_location='cpu'
        model = MobileFaceNet([112, 112],136)   
        checkpoint = torch.load(ckpt_path, map_location=map_location)      
        model.load_state_dict(checkpoint['state_dict'])
        return model
        
    def forward(self, im):
        # Landmark
        ldmk, _ = self.ldmk_detector(im)
        ldmk = ldmk.reshape(-1,68,2)
        
        # Facial 5 Points (x,y)
        lefteye = ldmk[:,36:42,:].mean(dim=1)
        righteye = ldmk[:,42:48,:].mean(dim=1)
        nose = ldmk[:,33,:]
        leftmouth = ldmk[:,48,:]
        rightmouth = ldmk[:,54,:]
        facial5points = torch.stack([righteye,lefteye,nose,rightmouth,leftmouth], dim=1)
        
        return ldmk, facial5points
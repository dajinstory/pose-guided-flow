import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from ..facial_recognition.model_irse import Backbone as Backbone_ID_Loss

def get_header(ch_in, ch_hidden, ch_out, kernel=3):
    pad = kernel // 2
    header = nn.Sequential(
        nn.Conv2d(ch_in, ch_hidden, kernel, padding=pad),
        nn.ReLU(),
        nn.Conv2d(ch_hidden, ch_hidden, kernel, padding=pad),
        nn.ReLU(),
        nn.Conv2d(ch_hidden, ch_out, kernel, padding=pad),
    )
    return header

def get_header2(ch_in, ch_hidden, ch_out, kernel=3):
    pad = kernel // 2
    header = nn.Sequential(
        nn.Conv2d(ch_in, ch_hidden, kernel, padding=pad),
        nn.ReLU(),
        nn.Conv2d(ch_hidden, ch_hidden, kernel, padding=pad),
        nn.ReLU(),
        nn.Conv2d(ch_hidden, ch_hidden, kernel, padding=pad),
        nn.ReLU(),
        nn.Conv2d(ch_hidden, ch_hidden, kernel, padding=pad),
        nn.ReLU(),
        nn.Conv2d(ch_hidden, ch_out, kernel, padding=pad),
    )
    return header

    
class VGG16Module(nn.Module):
    def __init__(self, pretrained=None):
        super(VGG16Module, self).__init__()
        
        # INPUT : (,3,32,32), resized to /2
        self.blocks = nn.Sequential(
            torchvision.models.vgg16(pretrained=True).features[:4].eval(),      # 64,H,W 
            torchvision.models.vgg16(pretrained=True).features[4:9].eval(),     # 128,/2,/2
            torchvision.models.vgg16(pretrained=True).features[9:16].eval(),    # 256,/4,/4
            torchvision.models.vgg16(pretrained=True).features[16:23].eval()   # 512,/8,/8
        )
        self.headers = nn.Sequential(
            get_header2(6,32,64,3),
            get_header2(12,64,128,3),
            get_header2(24,128,256,3),
            get_header2(48,256,512,3),            
        )
        self.require_resize = True
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]
        self.preprocess = T.Normalize(
            mean=self.norm_mean, 
            std=self.norm_std)
    
    def forward(self):
        pass

class InsightFaceModule(nn.Module):
    def __init__(self, pretrained, pretrained_headers=None):
        super(InsightFaceModule, self).__init__()

        facenet = Backbone_ID_Loss(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')  
        facenet.load_state_dict(torch.load(pretrained['ckpt_path']))  
        self.blocks = nn.Sequential(
            nn.Sequential(
                facenet.input_layer,
                facenet.body[:3]),      # 64,/2,/2
            facenet.body[3:7],          # 128,/4,/4
            facenet.body[7:21],         # 256,8,8
            facenet.body[21:],        # 512,4,4
        )
        self.headers = nn.Sequential(
            get_header2(6,32,64,3),
            get_header2(12,64,128,3),
            get_header2(24,128,256,3),
            get_header2(48,256,512,3),            
        )
        if pretrained_headers is not None:
            self.headers.load_state_dict(torch.load(pretrained_headers['ckpt_path']))  
        self.require_resize = False
        self.norm_mean = [0.5,0.5,0.5]
        self.norm_std = [0.5,0.5,0.5]
        self.preprocess = T.Normalize(
            mean=self.norm_mean, 
            std=self.norm_std)
            
    def forward(self):
        pass
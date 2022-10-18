import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from ..facial_recognition.model_irse import Backbone as Backbone_ID_Loss

def get_shallow_net1(ch_in, ch_hidden, ch_out, kernel=3):
    pad = kernel // 2
    header = nn.Sequential(
        nn.Conv2d(ch_in, ch_hidden, kernel, padding=pad),
        nn.ReLU(),
        nn.Conv2d(ch_hidden, ch_hidden, kernel, padding=pad),
        nn.ReLU(),
        nn.Conv2d(ch_hidden, ch_out, kernel, padding=pad),
    )
    return header

def get_shallow_net2(ch_in, ch_hidden, ch_out, kernel=3):
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

def get_vgg_header(ch_in, ch_hidden, ch_out, kernel=3):
    pad = kernel // 2
    header = nn.Sequential(
        get_shallow_net2(ch_in, ch_hidden, ch_out, kernel),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), # Follow vgg16 implements
    )
    return header

def get_insightface_header(ch_in, ch_hidden, ch_out, kernel=3):
    pad = kernel // 2
    header = nn.Sequential(
        get_shallow_net2(ch_in, ch_hidden, ch_out, kernel),
        nn.Sigmoid(),
    )
    return header

# def sub_conv(ch_in, ch_out):
#     return nn.Sequential(
#         nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
#         nn.ReLU(),
#         nn.Conv2d(ch_out, ch_out, kernel_size=1, padding=0),
#         nn.ReLU(),
#     )
                                    
class GlobalHeader(nn.Module):
    def __init__(self, in_size=64):
        super(GlobalHeader, self).__init__()
        
        self.in_size = in_size
        self.net = self._build_net(in_size)
    
    def _build_net(self, in_size):
        if in_size == 64:
            net = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 128, kernel_size=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False), 
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 32, kernel_size=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False), 
            )
        elif in_size == 256:
            net = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False), 
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False), 
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False), 
                nn.Conv2d(64, 32, kernel_size=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False), 
            )
        else:
            raise NotImplementedError(f'Global header not implemented with input size {in_size}')

        return net

    
    def forward(self, feature, out_size):
        global_feature = self.net(feature)
        global_feature = torch.cat([global_feature]*out_size, dim=2)
        global_feature = torch.cat([global_feature]*out_size, dim=3)

class VGG16Module(nn.Module):
    def __init__(self, pretrained=None):
        super(VGG16Module, self).__init__()
        
        self.blocks = nn.Sequential(
            torchvision.models.vgg16(pretrained=True).features[:4].eval(),      # 64,/2,/2 
            torchvision.models.vgg16(pretrained=True).features[4:9].eval(),     # 128,/4,/4
            torchvision.models.vgg16(pretrained=True).features[9:16].eval(),    # 256,/8,/8
            torchvision.models.vgg16(pretrained=True).features[16:23].eval()   # 512,/16,/16
        )
        # self.headers = nn.Sequential(
        #     get_vgg_header(12,64,64,3),
        #     get_vgg_header(48,128,128,3),
        #     get_vgg_header(192,256,256,3),
        #     get_vgg_header(768,512,512,3),            
        # )
        self.headers = nn.Sequential(
            get_vgg_header(6,32,64,3),
            get_vgg_header(12,64,128,3),
            get_vgg_header(24,128,256,3),
            get_vgg_header(48,256,512,3),            
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
            facenet.body[7:21],         # 256,/8,/8
            facenet.body[21:],          # 512,/16,/16
        )
        # self.headers = nn.Sequential(
        #     get_insightface_header(12,64,64,3),
        #     get_insightface_header(48,128,128,3),
        #     get_insightface_header(192,256,256,3),
        #     get_insightface_header(768,512,512,3),            
        # )
        self.headers = nn.Sequential(
            get_insightface_header(6,32,64,3),
            get_insightface_header(12,64,128,3),
            get_insightface_header(24,128,256,3),
            get_insightface_header(48,256,512,3),            
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
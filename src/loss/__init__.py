# Copyright (c) OpenMMLab. All rights reserved.
from .nll_loss import NLLLoss
from .pixelwise_loss import MSELoss, L1Loss
from .perceptual_loss import PerceptualLoss, PerceptualVGG  
from .gan_loss import GANLoss
from .contrastive_loss import TripletLoss, QuadrupletLoss
from .id_loss import IDLoss

__all__ = [
    'NLLLoss', 
    'MSELoss', 'L1Loss', 
    'TripletLoss', 'QuadrupletLoss',
    'PerceptualLoss', 'PerceptualVGG', 
    'GANLoss', 
    'IDLoss'
]

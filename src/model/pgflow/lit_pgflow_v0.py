import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.utils import make_grid

from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from ..common.lit_basemodel import LitBaseModel
from .pgflow_v0 import PGFlowV0
from .vgg_header import get_vgg_header
from loss import NLLLoss, TripletLoss, MSELoss, L1Loss, PerceptualLoss, IDLoss, GANLoss
from metric import L1, PSNR, SSIM

import os
import numbers
import numpy as np
from PIL import Image
from collections import OrderedDict

import cv2

# NLL + Triplet + RECON
class LitPGFlowV0(LitBaseModel):
    def __init__(self,
                 opt: dict,
                 pretrained=None,
                 strict_load=True):

        super().__init__()

        # network
        flow_nets = {
            'PGFlowV0': PGFlowV0,
        }

        self.opt = opt
        self.flow_net = flow_nets[opt['flow_net']['type']](**opt['flow_net']['args'])
        self.in_size = self.opt['in_size']
        self.n_bits = self.opt['n_bits']
        self.n_bins = 2.0**self.n_bits

        self.vgg_blocks = nn.Sequential(
            torchvision.models.vgg16(pretrained=True).features[:4].eval(),      # 64,64,64 
            torchvision.models.vgg16(pretrained=True).features[4:9].eval(),     # 128,32,32
            torchvision.models.vgg16(pretrained=True).features[9:16].eval(),    # 256,16,16
            torchvision.models.vgg16(pretrained=True).features[16:23].eval())   # 512,8,8
        self.vgg_headers = nn.Sequential(
            get_vgg_header(12,32,64,3),
            get_vgg_header(48,64,128,3),
            get_vgg_header(192,256,256,3),
            get_vgg_header(768,512,512,3),            
        )
        # self.vgg_headers = nn.Sequential(
        #     get_vgg_header(6,32,64,3),
        #     get_vgg_header(12,64,128,3),
        #     get_vgg_header(24,128,256,3),
        #     get_vgg_header(48,256,512,3),            
        # )

        self.norm_mean = [0.5, 0.5, 0.5]
        self.norm_std = [1.0, 1.0, 1.0] #[0.5, 0.5, 0.5]
        self.vgg_norm_mean = [0.485, 0.456, 0.406]
        self.vgg_norm_std = [0.229, 0.224, 0.225]
                
        self.preprocess = transforms.Normalize(
            mean=self.norm_mean, 
            std=self.norm_std)
        self.reverse_preprocess = transforms.Normalize(
            mean=[-m/s for m,s in zip(self.norm_mean, self.norm_std)],
            std=[1/s for s in self.norm_std])

        self.vgg_preprocess = transforms.Normalize(
            mean=self.vgg_norm_mean, 
            std=self.vgg_norm_std)

        # loss
        self._create_loss(opt['loss'])
        
        # metric
        self.sampled_images = []
        
        # log
        self.save_hyperparameters(ignore=[])

        # pretrained
        self.pretrained = pretrained
        
    def forward(self, x):
        pass

    def preprocess_batch(self, batch):
        # Data Quantization
        im, ldmks = batch
        im = torch.cat(im, dim=0)
        ldmks = [torch.cat(ldmk, dim=0) for ldmk in zip(*ldmks)]   
        im = im * 255

        if self.n_bits < 8:
            im = torch.floor(im / 2 ** (8 - self.n_bits))
        im = im / self.n_bins

        # Image preprocess
        im_resized = T.Resize(self.in_size//2, interpolation=InterpolationMode.BICUBIC, antialias=True)(im)
        im = self.preprocess(im)

        # VGG Guidance
        vgg_features = []
        with torch.no_grad():
            feature = self.vgg_preprocess(im_resized)
            for block in self.vgg_blocks:
                feature = block(feature)
                vgg_features.append(feature)

        # Conditions for affine-coupling layers
        conditions = ldmks[1:7]

        return im, conditions, vgg_features

    def training_step(self, batch, batch_idx):
        im, conditions, vgg_features = self.preprocess_batch(batch)

        # Forward
        quant_randomness = self.preprocess(torch.rand_like(im)/self.n_bins) - self.preprocess(torch.zeros_like(im)) # x = (0~1)/n_bins, \ (im-m)/s + (x-m)/s - (0-m)/s = (im+x-m)/s
        w, log_p, log_det, splits, inter_features = self.flow_net.forward(im + quant_randomness, conditions)
        inter_features = [ vgg_header(inter_feature) for vgg_header, inter_feature in zip(self.vgg_headers, inter_features[:4]) ]
        
        # Reverse - Latent to Image
        splits_random = [torch.randn_like(split) if split is not None else None for split in splits]
        splits_random_temp = [0.7*split if split is not None else None for split in splits_random]  
        w_s, conditions_s, im_s = self._prepare_self(w, conditions, im)
        # w_c, conditions_c, im_c = self._prepare_cross(w, conditions, im)
        # w_r, conditions_r, im_r = self._prepare_random(w, conditions, im)
        im_recs = self.flow_net.reverse(w_s, conditions_s, splits_random)
        # im_recc = self.flow_net.reverse(w_c, conditions_c)
        # im_recr = self.flow_net.reverse(w_r, conditions_r)
        
        # Clamp outputs
        im_recs = torch.clamp(self.reverse_preprocess(im_recs), 0, 1)
        # im_recc = torch.clamp(self.reverse_preprocess(im_recc), 0, 1)
        # im_recr = torch.clamp(self.reverse_preprocess(im_recr), 0, 1)
        im_s = torch.clamp(self.reverse_preprocess(im_s), 0, 1)
        # im_c = torch.clamp(self.reverse_preprocess(im_c), 0, 1)
        # im_r = torch.clamp(self.reverse_preprocess(im_r), 0, 1)
        
        # Loss
        losses = dict()
        losses['loss_nll'], log_nll = self.loss_nll(log_p, log_det, n_pixel=3*self.in_size*self.in_size)
        losses['loss_fg0'], log_fg0 = self.loss_fg(inter_features[0], vgg_features[0], weight=self.loss_fg_weights[0])
        losses['loss_fg1'], log_fg1 = self.loss_fg(inter_features[1], vgg_features[1], weight=self.loss_fg_weights[1])
        losses['loss_fg2'], log_fg2 = self.loss_fg(inter_features[2], vgg_features[2], weight=self.loss_fg_weights[2])
        losses['loss_fg3'], log_fg3 = self.loss_fg(inter_features[3], vgg_features[3], weight=self.loss_fg_weights[3])
        losses['loss_cvg'], log_cvg = self.loss_cvg(*torch.chunk(w, chunks=3, dim=0))
        losses['loss_recs'], log_recs = self.loss_recs(im_recs, im_s)
        # losses['loss_recc'], log_recc = self.loss_recc(im_recc, im_c)
        # losses['loss_recr'], log_recr = self.loss_recr(im_recr, im_r)
        loss_total_common = sum(losses.values())
        
        log_train = {
            'train/loss_nll': log_nll,
            'train/loss_fg0': log_fg0,
            'train/loss_fg1': log_fg1,
            'train/loss_fg2': log_fg2,
            'train/loss_fg3': log_fg3,
            'train/loss_cvg': log_cvg[0],
            'train/d_pos': log_cvg[1],
            'train/d_neg': log_cvg[2],
            'train/loss_recs': log_recs,
            # 'train/loss_recc': log_recc,
            # 'train/loss_recr': log_recr,
            'train/loss_total_common': loss_total_common,
        }
        
        # Log
        self.log_dict(log_train, logger=True, prog_bar=True)
        
        # Total Loss
        return loss_total_common


    def validation_step(self, batch, batch_idx):
        im, conditions, vgg_features = self.preprocess_batch(batch)

        # Forward
        w, log_p, log_det, splits, inter_features = self.flow_net.forward(im, conditions)
        inter_features = [ vgg_header(inter_feature) for vgg_header, inter_feature in zip(self.vgg_headers, inter_features) ]
        
        # Reverse - Latent to Image
        splits_random = [torch.randn_like(split) if split is not None else None for split in splits]
        splits_random_temp = [0.7*split if split is not None else None for split in splits_random]  
        w_s, conditions_s, im_s = self._prepare_self(w, conditions, im, stage='valid')
        w_c, conditions_c, im_c = self._prepare_cross(w, conditions, im)
        
        im_recs = self.flow_net.reverse(w_s, conditions_s, splits_random)
        im_recc = self.flow_net.reverse(w_c, conditions_c, splits_random)
        
        # Format - range (0~1)
        input = torch.clamp(self.reverse_preprocess(im_s), 0, 1)
        recon = torch.clamp(self.reverse_preprocess(im_recs), 0, 1)
        output = torch.clamp(self.reverse_preprocess(im_recc), 0, 1)
        gt = torch.clamp(self.reverse_preprocess(im_c), 0, 1)
        
        # Metric - Image, CHW
        if batch_idx < 10:
            self.sampled_images.append(input[0].cpu())
            self.sampled_images.append(recon[0].cpu())
            self.sampled_images.append(output[0].cpu())
            self.sampled_images.append(gt[0].cpu())
            
        # Metric - PSNR, SSIM
        input = input[0].cpu().numpy().transpose(1,2,0)
        recon = recon[0].cpu().numpy().transpose(1,2,0)
        output = output[0].cpu().numpy().transpose(1,2,0)
        gt = gt[0].cpu().numpy().transpose(1,2,0)
        
        metric_psnr = PSNR(output*255, gt*255) 
        metric_ssim = SSIM(output*255, gt*255)
        metric_l1 = L1(output*255, gt*255)

        log_valid = {
            'val/metric/psnr': metric_psnr,
            'val/metric/ssim': metric_ssim,
            'val/metric/l1': metric_l1}
        self.log_dict(log_valid)

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        grid = make_grid(self.sampled_images, nrow=4)
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_image(
                f'val/visualization',
                grid, self.global_step+1, dataformats='CHW')
        self.sampled_images = []

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        trainable_parameters = [*self.flow_net.parameters(), ]

        optimizer = Adam(
            trainable_parameters, 
            lr=self.opt['optim']['lr'], 
            betas=self.opt['optim']['betas'])
    
        scheduler = {
            'scheduler': CosineAnnealingLR(
                optimizer, 
                T_max=self.opt['scheduler']['T_max'], 
                eta_min=self.opt['scheduler']['eta_min']),
            'name': 'learning_rate_g'}
        

        return [optimizer], [scheduler]
    
    def _create_loss(self, opt):
        losses = {
            'NLLLoss': NLLLoss,
            'TripletLoss': TripletLoss,
            'MSELoss': MSELoss,
            'L1Loss': L1Loss,
            'PerceptualLoss': PerceptualLoss,
            'IDLoss': IDLoss,
            'GANLoss': GANLoss
        }
        
        self.loss_nll = losses[opt['nll']['type']](**opt['nll']['args'])
        self.loss_fg = losses[opt['feature_guide']['type']](**opt['feature_guide']['args'])
        self.loss_fg_weights = [1.0, 1.0, 1.0, 1.0]
        self.loss_cvg = losses[opt['cvg']['type']](**opt['cvg']['args'])
        self.loss_recs = losses[opt['recon_self']['type']](**opt['recon_self']['args'])
        # self.loss_recc = losses[opt['recon_cross']['type']](**opt['recon_cross']['args'])
        # self.loss_recr = losses[opt['recon_random']['type']](**opt['recon_random']['args'])

    def _prepare_self(self, w, conditions, im, stage='train'):
        n_batch = w.shape[0]//3
        w_ = w.clone().detach()[:2*n_batch]
        if stage=='valid':
            w_ = w_
        else:
            w_ = (w_[:n_batch] + w_[n_batch:2*n_batch]) / 2
            w_ = torch.cat([w_, w_], dim=0)   
        conditions_ = []
        for condition in conditions:                    
            conditions_.append(condition[:2*n_batch])
        im_ = im[:2*n_batch]
        return w_, conditions_, im_

    def _prepare_cross(self, w, conditions, im):
        n_batch = w.shape[0]//3
        w_ = w.clone().detach()[:2*n_batch]
        conditions_ = []
        for condition in conditions:
            conditions_.append(torch.cat([condition[n_batch:2*n_batch], condition[:n_batch]], dim=0))
        im_ = torch.cat([im[n_batch:2*n_batch], im[:n_batch]], dim=0)
        return w_, conditions_, im_

    def _prepare_random(self, w, conditions, im):
        n_batch = w.shape[0]//3
        w_ = torch.randn_like(w)[:n_batch] * 1.0
        conditions_ = []
        for condition in conditions:
            conditions_.append(condition[:n_batch])
        im_ = im[:n_batch]
        return w_, conditions_, im_

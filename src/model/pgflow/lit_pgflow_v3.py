import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.utils import make_grid

from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from ..common.lit_basemodel import LitBaseModel
from .pgflow_v2 import PGFlowV2
from .vgg_header import get_vgg_header
from util import floor, round
from loss import NLLLoss, TripletLoss, MSELoss, L1Loss, PerceptualLoss, IDLoss, GANLoss
from metric import L1, PSNR, SSIM

import os
import numbers
import numpy as np
from PIL import Image
from collections import OrderedDict

import cv2

# NLL, SPLIT, temp=0, for advanced model
class LitPGFlowV3(LitBaseModel):
    def __init__(self,
                 opt: dict,
                 pretrained=None,
                 strict_load=True):

        super().__init__()

        # network
        flow_nets = {
            'PGFlowV2': PGFlowV2,
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
            get_vgg_header(6,32,64,3),
            get_vgg_header(12,64,128,3),
            get_vgg_header(24,128,256,3),
            get_vgg_header(48,256,512,3),            
        )

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
    
    def preprocess_quant(self, im, option='round'):
        # floor
        if option == 'floor':
            im = im * 255
            if self.n_bits < 8:
                im = floor(im / 2 ** (8 - self.n_bits))
            im = im / self.n_bins

        # round
        elif option == 'round':
            im = im * 255
            if self.n_bits < 8:
                im = round(im / 2 ** (8 - self.n_bits))
            im = im / self.n_bins
        return im

    def preprocess_batch(self, batch):
        # Data Quantization
        im, ldmks = batch
        im = torch.cat(im, dim=0)
        ldmks = [torch.cat(ldmk, dim=0) for ldmk in zip(*ldmks)]   
        im = self.preprocess_quant(im)

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
        quant_randomness = self.preprocess(torch.rand_like(im)/self.n_bins - 0.5) - self.preprocess(torch.zeros_like(im)) # x = (-0.5~0.5)/n_bins, \ (im-m)/s + (x-m)/s - (0-m)/s = (im+x-m)/s
        w, log_p, log_det, splits, inter_features = self.flow_net.forward(im + quant_randomness, conditions)
        inter_features = [ vgg_header(inter_feature) for vgg_header, inter_feature in zip(self.vgg_headers, inter_features[:3]) ]
        
        # Reverse
        # w_s, conditions_s, splits_s, im_s = self._prepare_self(w, conditions, splits, im)
        # w_c, conditions_c, splits_c, im_c = self._prepare_cross(w, conditions, splits, im)
        w_m, conditions_m, splits_m, im_m = self._prepare_mean(w, conditions, splits, im)
        w_r, conditions_r, splits_r, im_r = self._prepare_random(w, conditions, splits, im)
        # im_recs = self.flow_net.reverse(w_s, conditions_s, splits_s)
        # im_recc = self.flow_net.reverse(w_c, conditions_c, splits_c)
        im_recm = self.flow_net.reverse(w_m, conditions_m, splits_m)
        im_recr = self.flow_net.reverse(w_r, conditions_r, splits_r)

        # Reverse_preprocess : -0.5~0.5 -> 0~1
        # im_recs = self.reverse_preprocess(im_recs)
        # im_recc = self.reverse_preprocess(im_recc)
        im_recm = self.reverse_preprocess(im_recm)
        im_recr = self.reverse_preprocess(im_recr)
        # im_s = self.reverse_preprocess(im_s)
        # im_c = self.reverse_preprocess(im_c)
        im_m = self.reverse_preprocess(im_m)
        im_r = self.reverse_preprocess(im_r)

        # Quantization
        # im_recs = self.preprocess_quant(im_recs)
        # im_recc = self.preprocess_quant(im_recc)
        im_recm = self.preprocess_quant(im_recm)
        im_recr = self.preprocess_quant(im_recr)
        
        # Clamp : (0,1)
        # im_recs = torch.clamp(im_recs, 0, 1)
        # im_recc = torch.clamp(im_recc, 0, 1)
        # im_recm = torch.clamp(im_recm, 0, 1)
        # im_recr = torch.clamp(im_recr, 0, 1)
        # im_s = torch.clamp(im_s, 0, 1)
        # im_c = torch.clamp(im_c, 0, 1)
        # im_m = torch.clamp(im_m, 0, 1)
        # im_r = torch.clamp(im_r, 0, 1)
        
        # Loss
        losses = dict()
        losses['loss_nll'], log_nll = self.loss_nll(log_p, log_det, n_pixel=3*self.in_size*self.in_size)
        # losses['loss_fg0'], log_fg0 = self.loss_fg(inter_features[0], vgg_features[0], weight=self.loss_fg_weights[0])
        # losses['loss_fg1'], log_fg1 = self.loss_fg(inter_features[1], vgg_features[1], weight=self.loss_fg_weights[1])
        # losses['loss_fg2'], log_fg2 = self.loss_fg(inter_features[2], vgg_features[2], weight=self.loss_fg_weights[2])
        # losses['loss_fg3'], log_fg3 = self.loss_fg(inter_features[3], vgg_features[3], weight=self.loss_fg_weights[3])
        losses['loss_cvg'], log_cvg = self.loss_cvg(*torch.chunk(w, chunks=3, dim=0))
        # losses['loss_recs'], log_recs = self.loss_recs(im_recs, im_s, weight= 0 if self.global_step < 0 else None)
        # losses['loss_recc'], log_recc = self.loss_recc(im_recc, im_c, weight= 0 if self.global_step < 0 else None)
        losses['loss_recm'], log_recm = self.loss_recm(im_recm, im_m, weight= 0 if self.global_step < 0 else None)
        losses['loss_recr'], log_recr = self.loss_recr(im_recr, im_r, weight= 0 if self.global_step < 0 else None)
        loss_total_common = sum(losses.values())
        
        log_train = {
            'train/loss_nll': log_nll,
            # 'train/loss_fg0': log_fg0,
            # 'train/loss_fg1': log_fg1,
            # 'train/loss_fg2': log_fg2,
            # 'train/loss_fg3': log_fg3,
            'train/loss_cvg': log_cvg[0],
            'train/d_pos': log_cvg[1],
            'train/d_neg': log_cvg[2],
            # 'train/loss_recs': log_recs,
            # 'train/loss_recc': log_recc,
            'train/loss_recm': log_recm,
            'train/loss_recr': log_recr,
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
        inter_features = [ vgg_header(inter_feature) for vgg_header, inter_feature in zip(self.vgg_headers, inter_features[:3]) ]
        
        # Reverse - Latent to Image
        w_s, conditions_s, splits_s, im_s = self._prepare_self(w, conditions, splits, im)
        w_c, conditions_c, splits_c, im_c = self._prepare_cross(w, conditions, splits, im)
        im_recs = self.flow_net.reverse(w_s, conditions_s, splits_s)
        im_recc = self.flow_net.reverse(w_c, conditions_c, splits_c)
        
        # Reverse_preprocess : -0.5~0.5 -> 0~1
        input = self.reverse_preprocess(im_s)
        recon = self.reverse_preprocess(im_recs)
        output = self.reverse_preprocess(im_recc)
        gt = self.reverse_preprocess(im_c)
        
        # Quantization
        input = self.preprocess_quant(input)
        recon = self.preprocess_quant(recon)
        output = self.preprocess_quant(output)
        gt = self.preprocess_quant(gt)

        # Clamp : (0,1)
        input = torch.clamp(input, 0, 1)
        recon = torch.clamp(recon, 0, 1)
        output = torch.clamp(output, 0, 1)
        gt = torch.clamp(gt, 0, 1)
        
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
        # Log Qualative Result - Image
        grid = make_grid(self.sampled_images, nrow=4)
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_image(
                f'val/visualization',
                grid, self.global_step+1, dataformats='CHW')
        self.sampled_images = []

        # # Update hyper-params if necessary
        # if self.current_epoch % 100 == 0:
        #     self.n_bits = min(self.n_bits+1, 8)
        #     self.n_bins = 2.0**self.n_bits
        #     self.loss_nll.n_bits = self.n_bits
        #     self.loss_nll.n_bins = self.n_bins

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
            'name': 'learning_rate'}
            
        # warmup = 5
        # lr_lambda = lambda epoch: min(1.0, (epoch + 1) / warmup)  # noqa
        # scheduler = {
        #     'scheduler': LambdaLR(
        #         optimizer, 
        #         lr_lambda=lr_lambda),
        #     'name': 'learning_rate'}

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
        self.loss_fg_weights = [0, 0, 0]
        # self.loss_fg_weights = [10, 0.5, 0.1] #[0.5, 0.2, 0.1, 0.2] #[0, 0, 0, 0] #[1.0, 0.5, 0.25, 0.125] #[0.01, 0.05, 0.1, 0.08]
        self.loss_cvg = losses[opt['cvg']['type']](**opt['cvg']['args'])
        self.loss_recs = losses[opt['recon_self']['type']](**opt['recon_self']['args'])
        self.loss_recc = losses[opt['recon_cross']['type']](**opt['recon_cross']['args'])
        self.loss_recm = losses[opt['recon_mean']['type']](**opt['recon_mean']['args'])
        self.loss_recr = losses[opt['recon_random']['type']](**opt['recon_random']['args'])

    def _prepare_self(self, w, conditions, splits, im, stage='train'):
        n_batch = w.shape[0]//3
        w_ = w.clone().detach()[:2*n_batch] 
        # splits_ = [split[:2*n_batch] if split is not None else None for split in splits]  
        # splits_ = [torch.randn_like(split)[:2*n_batch] if split is not None else None for split in splits]  
        splits_ = [torch.zeros_like(split)[:2*n_batch] if split is not None else None for split in splits]  
        conditions_ = []
        for condition in conditions:                    
            conditions_.append(condition[:2*n_batch])
        im_ = im[:2*n_batch]
        return w_, conditions_, splits_, im_

    def _prepare_cross(self, w, conditions, splits, im, stage='train'):
        n_batch = w.shape[0]//3
        w_ = w.clone().detach()[:2*n_batch]
        # splits_ = [torch.randn_like(split)[:2*n_batch] if split is not None else None for split in splits]  
        splits_ = [torch.zeros_like(split)[:2*n_batch] if split is not None else None for split in splits]  
        conditions_ = []
        for condition in conditions:
            conditions_.append(torch.cat([condition[n_batch:2*n_batch], condition[:n_batch]], dim=0))
        im_ = torch.cat([im[n_batch:2*n_batch], im[:n_batch]], dim=0)
        return w_, conditions_, splits_, im_

    def _prepare_mean(self, w, conditions, splits, im, stage='train'):
        n_batch = w.shape[0]//3
        w_ = w.clone().detach()[:2*n_batch] 
        w_ = (w_[:n_batch] + w_[n_batch:2*n_batch])/2
        w_ = torch.cat([w_, w_], dim=0)
        splits_ = [torch.zeros_like(split)[:2*n_batch] if split is not None else None for split in splits]  
        conditions_ = []
        for condition in conditions:                    
            conditions_.append(condition[:2*n_batch])
        im_ = im[:2*n_batch]
        return w_, conditions_, splits_, im_

    def _prepare_random(self, w, conditions, splits, im, stage='train'):
        n_batch = w.shape[0]//3
        w_ = w.clone().detach()[:2*n_batch] 
        w_ = torch.randn_like(w_)
        splits_ = [torch.zeros_like(split)[:2*n_batch] if split is not None else None for split in splits]  
        conditions_ = []
        for condition in conditions:                    
            conditions_.append(condition[:2*n_batch])
        im_ = im[:2*n_batch]
        return w_, conditions_, splits_, im_
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.utils import make_grid

from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from ..common.lit_basemodel import LitBaseModel
from ..landmark_detector.landmark_detector import FacialLandmarkDetector
from .pgflow_v3 import PGFlowV3
from .module import VGG16Module, InsightFaceModule, GlobalHeader
from util import floor, round
from loss import NLLLoss, TripletLoss, QuadrupletLoss, MSELoss, L1Loss, PerceptualLoss, IDLoss, GANLoss
from metric import L1, PSNR, SSIM

import os
import numbers
import numpy as np
from PIL import Image
from collections import OrderedDict
import cv2

class LitPGFlowV3(LitBaseModel):
    def __init__(self,
                 opt: dict,
                 pretrained=None,
                 strict_load=True):

        super().__init__()

        # opt
        self.opt = opt
        if pretrained is True:
            self.opt['flow_net']['args']['pretrained'] = True
            
        # network
        flow_nets = {
            'PGFlowV3': PGFlowV3,
        }
        ldmk_detectors = {
            'FacialLandmarkDetector': FacialLandmarkDetector,
        }
        kd_modules = {
            'VGG16Module': VGG16Module,
            'InsightFaceModule': InsightFaceModule,
        }

        self.flow_net = flow_nets[opt['flow_net']['type']](**opt['flow_net']['args'])
        self.in_size = self.opt['in_size']
        self.n_bits = self.opt['n_bits']
        self.n_bins = 2.0**self.n_bits

        self.ldmk_detector = ldmk_detectors[opt['landmark_detector']['type']](**opt['landmark_detector']['args'])

        self.kd_module = kd_modules[opt['kd_module']['type']](**opt['kd_module']['args'])
        # self.global_header = GlobalHeader(in_size=self.in_size)

        self.norm_mean = [0.5, 0.5, 0.5]
        self.norm_std = [1.0, 1.0, 1.0]
                
        self.preprocess = T.Normalize(
            mean=self.norm_mean, 
            std=self.norm_std)
        self.reverse_preprocess = T.Normalize(
            mean=[-m/s for m,s in zip(self.norm_mean, self.norm_std)],
            std=[1/s for s in self.norm_std])

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
        # Batch
        im, conditions, ldmk, f5p = batch
        im = torch.cat(im, dim=0)
        conditions = [torch.cat(condition, dim=0) for condition in zip(*conditions)]   
        ldmk = torch.cat(ldmk, dim=0)
        f5p = torch.cat(f5p, dim=0)

        # Im_resized for VGG Header
        im_resized = T.Resize(self.in_size//2, interpolation=InterpolationMode.BICUBIC, antialias=True)(im)

        # Data Quantization, (0,1)
        im = self.preprocess_quant(im)
        im_resized = self.preprocess_quant(im_resized)        

        # KD Guidance
        kd_features = []
        self.kd_module.blocks.eval()
        with torch.no_grad():
            if type(self.kd_module) is VGG16Module:
                feature = self.kd_module.preprocess(im_resized) # VGG16 : input: norm( (0,1) )
            elif type(self.kd_module) is InsightFaceModule:
                feature = self.kd_module.preprocess(im) # InsightFace : input: norm( (0,1) )
            else:
                raise ValueError(f'KD-Module [{type(self.kd_module)}] is not supported (Preprocessing)')

            for block in self.kd_module.blocks:
                feature = block(feature)
                kd_features.append(feature)

        # Global Feature
        # global_feature = self.global_header(kd_features[-1], out_size=self.in_size)
        global_feature = None
        
        # Preprocess Inputs
        im = self.preprocess(im)
        conditions = [global_feature] + conditions[1:7]
        kd_features = kd_features[:]

        return im, conditions, kd_features, ldmk, f5p

    def training_step(self, batch, batch_idx):
        im, conditions, kd_features, ldmk, f5p = self.preprocess_batch(batch)
        n_batch = im.shape[0]//4

        # Forward
        quant_randomness = self.preprocess(torch.rand_like(im)/self.n_bins - 0.5) - self.preprocess(torch.zeros_like(im)) # x = (-0.5~0.5)/n_bins, \ (im-m)/s + (x-m)/s - (0-m)/s = (im+x-m)/s
        # im = im + quant_randomness
        w, log_p, log_det, splits, inter_features = self.flow_net.forward(im, conditions)
        inter_features = [ kd_header(inter_feature) for kd_header, inter_feature in zip(self.kd_module.headers[0:4], inter_features[1:5]) ]

        # Reverse_function
        def compute_im_recon(w, conditions, splits, im):
            # Flow.reverse
            im_rec = self.flow_net.reverse(w, conditions, splits)
            # Range : (-0.5, 0.5) -> (0,1)
            im_rec = self.reverse_preprocess(im_rec)
            im = self.reverse_preprocess(im)
            # Quantization
            # im_rec = self.preprocess_quant(im_rec)
            # Clamp : (0,1)
            # im_rec = torch.clamp(im_rec, 0, 1)
            # im = torch.clamp(im, 0, 1)
            return im_rec, im
        
        # Reverse
        w_s, conditions_s, splits_s, im_s, ldmk_s, f5p_s = self._prepare_self(w, conditions, splits, im, ldmk, f5p)
        w_c, conditions_c, splits_c, im_c, ldmk_c, f5p_c = self._prepare_cross(w, conditions, splits, im, ldmk, f5p)
        im_recs, im_s = compute_im_recon(w_s, conditions_s, splits_s, im_s)
        im_recc, im_c = compute_im_recon(w_c, conditions_c, splits_c, im_c)
        
        # Pose
        # im_recc_resized = T.Resize(112, interpolation=InterpolationMode.BICUBIC, antialias=True)(im_recc)
        # ldmk_recc, f5p_recc = self.ldmk_detector(im_recc_resized) # input: (0,1)

        # Loss
        losses = dict()
        losses['loss_nll'], log_nll = self.loss_nll(log_p, log_det, n_pixel=3*self.in_size*self.in_size)
        losses['loss_fg0'], log_fg0 = self.loss_fg(inter_features[0], kd_features[0])
        losses['loss_fg1'], log_fg1 = self.loss_fg(inter_features[1], kd_features[1])
        losses['loss_fg2'], log_fg2 = self.loss_fg(inter_features[2], kd_features[2])
        losses['loss_fg3'], log_fg3 = self.loss_fg(inter_features[3], kd_features[3])
        losses['loss_cvg'], log_cvg = self.loss_cvg(*torch.chunk(w, chunks=4, dim=0))
        losses['loss_recs'], log_recs = self.loss_recs(im_recs, im_s, weight= 0 if self.global_step < 0 else None)
        losses['loss_recc'], log_recc = self.loss_recc(im_recc, im_c, weight= 0 if self.global_step < 0 else None)
        (losses['loss_perc'], losses['loss_stlc']), (log_perc, log_stlc) = self.loss_perc(im_recc, im_c)
        losses['loss_idc'], log_idc = self.loss_idc(im_recc, im_c)        
        # losses['loss_ldmk'], log_ldmk = self.loss_ldmk(ldmk_recc, ldmk_c, weight= 0 if self.global_step < 0 else None)
        # losses['loss_f5p'], log_f5p = self.loss_f5p(f5p_recc, f5p_c, weight= 0 if self.global_step < 0 else None)
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
            'train/loss_recc': log_recc,
            'train/loss_perc': log_perc,
            'train/loss_stlc': log_stlc,
            'train/loss_idc': log_idc,
            # 'train/loss_ldmk': log_ldmk,
            # 'train/loss_f5p': log_f5p,
            'train/loss_total_common': loss_total_common,
        }
        
        # Log
        self.log_dict(log_train, logger=True, prog_bar=True)
        
        # Total Loss
        return loss_total_common


    def validation_step(self, batch, batch_idx):
        # if batch_idx == 0:
        #     torch.save(self.flow_net.state_dict(), 'flow.ckpt')
        #     torch.save(self.global_header.state_dict(), 'global_header.ckpt')
        #     torch.save(self.kd_module.headers.state_dict(), 'kd_module_headers.ckpt')

        im, conditions, kd_features, ldmk, f5p = self.preprocess_batch(batch)

        # Forward
        w, log_p, log_det, splits, inter_features = self.flow_net.forward(im, conditions)
        inter_features = [ kd_header(inter_feature) for kd_header, inter_feature in zip(self.kd_module.headers[0:4], inter_features[1:5]) ]
        
        # Reverse - Latent to Image
        w_s, conditions_s, splits_s, im_s, ldmk_s, f5p_s = self._prepare_self(w, conditions, splits, im, ldmk, f5p, stage='valid')
        w_c, conditions_c, splits_c, im_c, ldmk_c, f5p_c = self._prepare_cross(w, conditions, splits, im, ldmk, f5p, stage='valid')
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
        # if batch_idx == 0:
        #     torch.save(self.flow_net.state_dict(), 'pgflow.ckpt')
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
        trainable_parameters = [
            *self.flow_net.parameters(), 
            *self.kd_module.headers.parameters(), 
            # *self.global_header.parameters(),
        ]

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
            
        return [optimizer], [scheduler]
    
    def _create_loss(self, opt):
        losses = {
            'NLLLoss': NLLLoss,
            'TripletLoss': TripletLoss,
            'QuadrupletLoss': QuadrupletLoss,
            'MSELoss': MSELoss,
            'L1Loss': L1Loss,
            'PerceptualLoss': PerceptualLoss,
            'IDLoss': IDLoss,
            'GANLoss': GANLoss
        }
        
        self.loss_nll = losses[opt['nll']['type']](**opt['nll']['args'])
        self.loss_fg = losses[opt['feature_guide']['type']](**opt['feature_guide']['args'])
        self.loss_cvg = losses[opt['cvg']['type']](**opt['cvg']['args'])
        self.loss_recs = losses[opt['recon_self']['type']](**opt['recon_self']['args'])
        self.loss_recc = losses[opt['recon_cross']['type']](**opt['recon_cross']['args'])
        self.loss_perc = losses[opt['perc_cross']['type']](**opt['perc_cross']['args'])
        self.loss_idc = losses[opt['id_cross']['type']](**opt['id_cross']['args'])
        self.loss_ldmk = losses[opt['landmark']['type']](**opt['landmark']['args'])
        self.loss_f5p = losses[opt['facial5points']['type']](**opt['facial5points']['args'])

    def _prepare_self(self, w, conditions, splits, im, ldmk, f5p, stage='train'):
        n_batch = w.shape[0]//4
        w_ = w
        temp = 0.7 if stage == 'train' else 0
        splits_ = [temp * torch.randn_like(split) * self.flow_net.inter_temp if split is not None else None for split in splits]  
        conditions_ = conditions
        im_ = im
        ldmk_ = ldmk
        f5p_ = f5p
        return w_, conditions_, splits_, im_, ldmk_, f5p_

    def _prepare_cross(self, w, conditions, splits, im, ldmk, f5p, stage='train'):
        n_batch = w.shape[0]//4
        w_ = w
        temp = 0.7 if stage == 'train' else 0
        splits_ = [temp * torch.randn_like(split) * self.flow_net.inter_temp if split is not None else None for split in splits]  
        conditions_ = [ torch.cat([ 
            condition[n_batch:2*n_batch], 
            condition[:n_batch],
            condition[3*n_batch:],
            condition[2*n_batch:3*n_batch]
            ], dim=0) if condition is not None else None for condition in conditions ]
        conditions_[0] = conditions[0]
        im_ = torch.cat([
            im[n_batch:2*n_batch], 
            im[:n_batch],
            im[3*n_batch:], 
            im[2*n_batch:3*n_batch],
            ], dim=0)
        ldmk_ = torch.cat([
            ldmk[n_batch:2*n_batch], 
            ldmk[:n_batch],
            ldmk[3*n_batch:], 
            ldmk[2*n_batch:3*n_batch]
            ], dim=0)
        f5p_ = torch.cat([
            f5p[n_batch:2*n_batch], 
            f5p[:n_batch],
            f5p[3*n_batch:], 
            f5p[2*n_batch:3*n_batch]
            ], dim=0)
        return w_, conditions_, splits_, im_, ldmk_, f5p_
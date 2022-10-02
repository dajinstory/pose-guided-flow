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
from .pgflow_v1 import PGFlowV1
from .pgflow_v2 import PGFlowV2
from .module import VGG16Module, InsightFaceModule
from util import computeGaussian, draw_edge
from util import floor, round
from loss import NLLLoss, TripletLoss, MSELoss, L1Loss, PerceptualLoss, IDLoss, GANLoss
from metric import L1, PSNR, SSIM

import os
import numbers
import numpy as np
from PIL import Image
from collections import OrderedDict
import cv2

ptt = T.ToTensor()
ttp = T.ToPILImage()

# NLL, SPLIT, temp=0, LDMK loss
class LitPGFlowV2(LitBaseModel):
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
            'PGFlowV1': PGFlowV1,
            'PGFlowV2': PGFlowV2,
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

        # self.kd_module = VGG16Module()
        self.kd_module = kd_modules[opt['kd_module']['type']](**opt['kd_module']['args'])

        self.norm_mean = [0.5, 0.5, 0.5]
        self.norm_std = [1.0, 1.0, 1.0] #[0.5, 0.5, 0.5]
                
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

        # Data Quantization, (0,1)
        im = self.preprocess_quant(im)

        # Landmarks      
        # im_resized = T.Resize(112, interpolation=InterpolationMode.BICUBIC, antialias=True)(im)
        # ldmk, facial5points = self.ldmk_detector(im_resized) # input: (0,1)
        # conditions = []
        # res = self.in_size
        # for _ in range(7):
        #     heatmap = []
        #     edgemap = []
        #     for i in range(ldmk.shape[0]):
        #         heatmap_i = computeGaussian(ldmk[i], res=res, kernel_sigma=0.1, device=ldmk[i].get_device())
        #         edgemap_i = ptt(draw_edge(ldmk[i], img_size=res)).to(ldmk[i].get_device())
        #         heatmap.append(heatmap_i)
        #         edgemap.append(edgemap_i)
        #     heatmap = torch.stack(heatmap, dim=0)
        #     edgemap = torch.stack(edgemap, dim=0)
        #     # print(heatmap.shape, edgemap.shape, flush=True)
        #     conditions.append(torch.cat([heatmap, edgemap], dim=1))
        #     res = res // 2

        # KD Guidance
        kd_features = []
        with torch.no_grad():
            feature = self.kd_module.preprocess(im) # input: norm( (0,1) )
            for block in self.kd_module.blocks:
                feature = block(feature)
                kd_features.append(feature)

        # Preprocess Inputs
        im = self.preprocess(im)
        conditions = conditions[1:7]#[1:5]
        kd_features = kd_features[:]

        return im, conditions, kd_features, ldmk, f5p

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            im, conditions, kd_features, ldmk, f5p = self.preprocess_batch(batch)
            n_batch = im.shape[0]//3

        # Forward
        quant_randomness = self.preprocess(torch.rand_like(im)/self.n_bins - 0.5) - self.preprocess(torch.zeros_like(im)) # x = (-0.5~0.5)/n_bins, \ (im-m)/s + (x-m)/s - (0-m)/s = (im+x-m)/s
        # im = im + quant_randomness
        w, log_p, log_det, splits, inter_features = self.flow_net.forward(im, conditions)
        inter_features = [ kd_header(inter_feature) for kd_header, inter_feature in zip(self.kd_module.headers, inter_features[:4]) ]

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
        # w_m, conditions_m, splits_m, im_m, ldmk_m, f5p_m = self._prepare_mean(w, conditions, splits, im, ldmk, f5p)
        # w_r, conditions_r, splits_r, im_r, ldmk_r, f5p_r = self._prepare_random(w, conditions, splits, im, ldmk, f5p)
        im_recs, im_s = compute_im_recon(w_s, conditions_s, splits_s, im_s)
        im_recc, im_c = compute_im_recon(w_c, conditions_c, splits_c, im_c)
        # im_recm, im_m = compute_im_recon(w_m, conditions_m, splits_m, im_m)
        # im_genr, _ = compute_im_recon(w_r, conditions_r, splits_r, im_r)
        
        # Pose
        # im_genr_resized = T.Resize(112, interpolation=InterpolationMode.BICUBIC, antialias=True)(im_genr)
        # ldmk_genr, f5p_genr = self.ldmk_detector(im_genr_resized) # input: (0,1)
        im_recc_resized = T.Resize(112, interpolation=InterpolationMode.BICUBIC, antialias=True)(im_recc)
        ldmk_recc, f5p_recc = self.ldmk_detector(im_recc_resized) # input: (0,1)

        # Loss
        losses = dict()
        losses['loss_nll'], log_nll = self.loss_nll(log_p, log_det, n_pixel=3*self.in_size*self.in_size)
        losses['loss_fg0'], log_fg0 = self.loss_fg(inter_features[0], kd_features[0])
        losses['loss_fg1'], log_fg1 = self.loss_fg(inter_features[1], kd_features[1])
        losses['loss_fg2'], log_fg2 = self.loss_fg(inter_features[2], kd_features[2])
        losses['loss_fg3'], log_fg3 = self.loss_fg(inter_features[3], kd_features[3])
        losses['loss_cvg'], log_cvg = self.loss_cvg(*torch.chunk(w, chunks=3, dim=0))
        losses['loss_recs'], log_recs = self.loss_recs(im_recs, im_s, weight= 0 if self.global_step < 0 else None)
        losses['loss_recc'], log_recc = self.loss_recc(im_recc, im_c, weight= 0 if self.global_step < 0 else None)
        # losses['loss_recm'], log_recm = self.loss_recm(im_recm, im_m, weight= 0 if self.global_step < 0 else None)
        losses['loss_ldmk'], log_ldmk = self.loss_ldmk(ldmk_recc, ldmk_c, weight= 0 if self.global_step < 0 else None)
        losses['loss_f5p'], log_f5p = self.loss_f5p(f5p_recc, f5p_c, weight= 0 if self.global_step < 0 else None)
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
            # 'train/loss_recm': log_recm,
            'train/loss_ldmk': log_ldmk,
            'train/loss_f5p': log_f5p,
            'train/loss_total_common': loss_total_common,
        }
        
        # Log
        self.log_dict(log_train, logger=True, prog_bar=True)
        
        # Total Loss
        return loss_total_common


    def validation_step(self, batch, batch_idx):
        # print(self.loss_nll.weight, flush=True)
        # print(self.loss_cvg.weight, flush=True)
        # print(self.loss_fg.weight, flush=True)
        # print(self.loss_recs.weight, flush=True)
        # print(self.loss_recc.weight, flush=True)
        # print(self.loss_ldmk.weight, flush=True)
        # print(self.loss_f5p.weight, flush=True)
        # if batch_idx == 0:
        #     torch.save(self.flow_net.state_dict(), 'pgflow.ckpt')
        #     torch.save(self.kd_module.headers.state_dict(), 'kd_headers.ckpt')

        with torch.no_grad():
            im, conditions, kd_features, ldmk, f5p = self.preprocess_batch(batch)

        # Forward
        w, log_p, log_det, splits, inter_features = self.flow_net.forward(im, conditions)
        inter_features = [ kd_header(inter_feature) for kd_header, inter_feature in zip(self.kd_module.headers, inter_features[:4]) ]
        
        # Reverse - Latent to Image
        w_s, conditions_s, splits_s, im_s, ldmk_s, f5p_s = self._prepare_self(w, conditions, splits, im, ldmk, f5p)
        w_c, conditions_c, splits_c, im_c, ldmk_c, f5p_c = self._prepare_cross(w, conditions, splits, im, ldmk, f5p)
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
        trainable_parameters = [*self.flow_net.parameters(), *self.kd_module.headers.parameters(),]

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
            'MSELoss': MSELoss,
            'L1Loss': L1Loss,
            'PerceptualLoss': PerceptualLoss,
            'IDLoss': IDLoss,
            'GANLoss': GANLoss
        }
        
        self.loss_nll = losses[opt['nll']['type']](**opt['nll']['args'])
        self.loss_fg = losses[opt['feature_guide']['type']](**opt['feature_guide']['args'])
        # self.loss_fg_weights = [10, 0.5, 0.1] #[0.5, 0.2, 0.1, 0.2] #[0, 0, 0, 0] #[1.0, 0.5, 0.25, 0.125] #[0.01, 0.05, 0.1, 0.08]
        self.loss_cvg = losses[opt['cvg']['type']](**opt['cvg']['args'])
        self.loss_recs = losses[opt['recon_self']['type']](**opt['recon_self']['args'])
        self.loss_recc = losses[opt['recon_cross']['type']](**opt['recon_cross']['args'])
        self.loss_recm = losses[opt['recon_mean']['type']](**opt['recon_mean']['args'])
        self.loss_ldmk = losses[opt['landmark']['type']](**opt['landmark']['args'])
        self.loss_f5p = losses[opt['facial5points']['type']](**opt['facial5points']['args'])

    def _prepare_self(self, w, conditions, splits, im, ldmk, f5p, stage='train'):
        n_batch = w.shape[0]//3
        # w_ = w.clone().detach()[:2*n_batch] 
        w_ = w[:2*n_batch]
        splits_ = [0.7 * torch.randn_like(split)[:2*n_batch] * self.flow_net.inter_temp if split is not None else None for split in splits]  
        # splits_ = [torch.zeros_like(split)[:2*n_batch] if split is not None else None for split in splits]  
        conditions_ = [condition[:2*n_batch] for condition in conditions]
        im_ = im[:2*n_batch]
        ldmk_ = ldmk[:2*n_batch]
        f5p_ = f5p[:2*n_batch]
        # return w_, conditions_, splits_, im_, ldmk_, f5p_
        return w, conditions, splits, im, ldmk, f5p

    def _prepare_cross(self, w, conditions, splits, im, ldmk, f5p, stage='train'):
        n_batch = w.shape[0]//3
        # w_ = w.clone().detach()[:2*n_batch]
        w_ = w[:2*n_batch] 
        splits_ = [0.7 * torch.randn_like(split)[:2*n_batch] * self.flow_net.inter_temp if split is not None else None for split in splits]  
        # splits_ = [torch.zeros_like(split)[:2*n_batch] if split is not None else None for split in splits]  
        conditions_ = []
        conditions_ = [torch.cat([condition[n_batch:2*n_batch], condition[:n_batch]], dim=0) for condition in conditions]
        im_ = torch.cat([im[n_batch:2*n_batch], im[:n_batch]], dim=0)
        ldmk_ = torch.cat([ldmk[n_batch:2*n_batch], ldmk[:n_batch]], dim=0)
        f5p_ = torch.cat([f5p[n_batch:2*n_batch], f5p[:n_batch]], dim=0)
        return w_, conditions_, splits_, im_, ldmk_, f5p_

    def _prepare_mean(self, w, conditions, splits, im, ldmk, f5p, stage='train'):
        n_batch = w.shape[0]//3
        # w_ = w.clone().detach()[:2*n_batch] 
        w_ = w[:2*n_batch] 
        w_ = (w_[:n_batch] + w_[n_batch:2*n_batch])/2
        w_ = torch.cat([w_, w_], dim=0)
        splits_ = [0.7 * torch.randn_like(split)[:2*n_batch] * self.flow_net.inter_temp if split is not None else None for split in splits]  
        # splits_ = [torch.zeros_like(split)[:2*n_batch] if split is not None else None for split in splits]  
        conditions_ = []
        for condition in conditions:                    
            conditions_.append(condition[:2*n_batch])
        im_ = im[:2*n_batch]
        ldmk_ = ldmk[:2*n_batch]
        f5p_ = f5p[:2*n_batch]
        return w_, conditions_, splits_, im_, ldmk_, f5p_

    def _prepare_random(self, w, conditions, splits, im, ldmk, f5p, stage='train'):
        n_batch = w.shape[0]//3
        w_ = torch.randn_like(w)[:2*n_batch]
        splits_ = [0.7 * torch.randn_like(split)[:2*n_batch] * self.flow_net.inter_temp if split is not None else None for split in splits]  
        # splits_ = [torch.zeros_like(split)[:2*n_batch] if split is not None else None for split in splits]  
        conditions_ = [condition[:2*n_batch] for condition in conditions]
        im_ = im[:2*n_batch]
        ldmk_ = ldmk[:2*n_batch]
        f5p_ = f5p[:2*n_batch]
        return w_, conditions_, splits_, im_, ldmk_, f5p_
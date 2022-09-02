import torch
import torch.nn as nn

from math import log, sqrt, pi, exp, cos, sin

from ..common.flow_module import gaussian_log_p
from ..common.flow_module import Block, FakeBlock, ZeroConv2d

def sub_conv(ch_hidden, kernel):
    pad = kernel // 2
    return lambda ch_in, ch_out: nn.Sequential(
                                    nn.Conv2d(ch_in, ch_hidden, kernel, padding=pad),
                                    nn.ReLU(),
                                    nn.Conv2d(ch_hidden, ch_hidden, kernel, padding=pad),
                                    nn.ReLU(),
                                    ZeroConv2d(ch_hidden, ch_out),)

class PGFlowV0(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()

        # configs
        self.img_size = 64
        self.n_levels = 4
        self.n_vectors = 4
        self.level_chunks = 4
        self.vector_chunks = 1
        self.inter_temp = 1.0
        self.final_temp = 1.0

        # Blocks (3,64,64) -> (768,4,4)
        self.blocks = nn.Sequential(
            Block(squeeze=True, # (12,32,32)
                  flow_type='InvConvFlow', n_flows=8, ch_in=12, ch_c=68, n_chunk=2, subnet=sub_conv(64,3), clamp=1.0, clamp_activation='GLOW',
                  split=False),
            Block(squeeze=True, # (48,16,16)
                  flow_type='InvConvFlow', n_flows=8, ch_in=48, ch_c=68, n_chunk=2, subnet=sub_conv(128,3), clamp=1.0, clamp_activation='GLOW',
                  split=False),
            Block(squeeze=True, # (192,8,8)
                  flow_type='InvConvFlow', n_flows=8, ch_in=192, ch_c=68, n_chunk=2, subnet=sub_conv(256,3), clamp=1.0, clamp_activation='GLOW',
                  split=False),
            Block(squeeze=True, # (768,4,4)
                  flow_type='InvConvFlow', n_flows=8, ch_in=768, ch_c=68, n_chunk=2, subnet=sub_conv(512,3), clamp=1.0, clamp_activation='GLOW',
                  split=False),
        )

        # Headers (b,768,4,4) -> (16xb,768,1,1)
        self.headers = nn.Sequential()
        for feature_level in range(self.n_levels): # 4x(b,192,4,4)
            self.headers.append(nn.Sequential())
            self.headers[feature_level].append(
                Block(squeeze=True, # (b,768,2,2)
                      flow_type='InvConvFlow', n_flows=4, ch_in=768, ch_c=68, n_chunk=2, subnet=sub_conv(512,3), clamp=1.0, clamp_activation='GLOW',
                      split=False)
            )
            self.headers[feature_level].append(
                nn.Sequential(
                    Block(squeeze=True, # 1st (b,768,1,1) 
                          flow_type='InvConvFlow', n_flows=4, ch_in=768, ch_c=68, n_chunk=2, subnet=sub_conv(512,3), clamp=1.0, clamp_activation='GLOW',
                          split=False),
                    Block(squeeze=True, # 2nd (b,768,1,1) 
                          flow_type='InvConvFlow', n_flows=4, ch_in=768, ch_c=68, n_chunk=2, subnet=sub_conv(512,3), clamp=1.0, clamp_activation='GLOW',
                          split=False),
                    Block(squeeze=True, # 3rd (b,768,1,1) 
                          flow_type='InvConvFlow', n_flows=4, ch_in=768, ch_c=68, n_chunk=2, subnet=sub_conv(512,3), clamp=1.0, clamp_activation='GLOW',
                          split=False),
                    Block(squeeze=True, # 4th (b,768,1,1) 
                          flow_type='InvConvFlow', n_flows=4, ch_in=768, ch_c=68, n_chunk=2, subnet=sub_conv(512,3), clamp=1.0, clamp_activation='GLOW',
                          split=False),
                )
            )

        # checkpoint
        if pretrained is not None:
            ckpt_path = pretrained['ckpt_path']
            print("Load flownet - Checkpoint : ", ckpt_path, flush=True)
            self.init_weights(ckpt_path)
        else:
            # print("Load flownet -  Initial Random N(0,0.01)", flush=True)
            # for p in self.parameters():
            #     p.data = 0.01 * torch.randn_like(p)
            print("Load flownet -  No Initialize", flush=True)
 
    def init_weights(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path), strict=True)
        for block in [*self.blocks, *self.headers]:
            for flow in block.flows:
                flow.actnorm.inited=True
        
    def forward(self, x, conditions):
        output = x        
        log_p = 0
        log_det = 0  

        # Blocks (3,64,64) -> (768,4,4)
        for block, condition in zip(self.blocks, conditions[:len(self.blocks)]):
            output, _log_det, _split = block(output, condition)
            log_det = log_det + _log_det

            if _split is not None:
                split = _split
                split = split.view(split.shape[0], -1)
                _m = torch.zeros_like(split)
                _log_sd = torch.ones_like(split) * log(self.inter_temp)
                _log_p = gaussian_log_p(split, _m, _log_sd)
                log_p += _log_p.sum(1)

        # Headers
        ## Format (b,c,h,w) -> 4x(b,c/4,h,w)
        b,c,h,w = output.shape
        output = output.view(b, 4, c//4, h, w)
        output = output.permute(1, 0, 2, 3, 4)
        
        output_by_levels = []
        for feature_level in range(self.n_levels): # ['layout', 'object', 'attribute', 'color']
            # 1st Layer
            condition = conditions[len(self.blocks)]
            output_by_level = output[feature_level]
            header_by_level = self.headers[feature_level][0]
            output_by_level, _log_det, _ = header_by_level(output_by_level, condition) # (b,768,_,_)
            log_det = log_det + _log_det
        
            # 2nd Layer
            _b,_c,_h,_w = output_by_level.shape
            output_by_level = output_by_level.view(_b, 4, _c//4, _h, _w)
            output_by_level = output_by_level.permute(1, 0, 2, 3, 4)
            output_by_vectors = []
            for vector_idx in range(self.n_vectors):
                condition = conditions[len(self.blocks)+1]
                output_by_vector = output_by_level[vector_idx]
                header_by_vector = self.headers[feature_level][1][vector_idx]
                output_by_vector, _log_det, _ = header_by_vector(output_by_vector, condition)
                log_det = log_det + _log_det
                output_by_vectors.append(output_by_vector)

            output_by_vectors = torch.stack(output_by_vectors, dim=1) # 4x(b,768,1,1) -> (b,4,768,1,1)
            output_by_vectors = output_by_vectors.view(-1,4,768)
            output_by_levels.append(output_by_vectors) # (b,4,768)

        outputs = torch.stack(output_by_levels, dim=1) # n_levels x (b,n_vectors,768) -> (b, n_levels, n_vectors, 768)
        outputs = outputs.view(-1,16,768)
        w = outputs

        # Calculate log_p for final Z
        z = outputs.view(outputs.shape[0], -1)
        _m = torch.zeros_like(z)
        _log_sd = torch.ones_like(z) * log(self.final_temp)
        _log_p = gaussian_log_p(z, _m, _log_sd)
        log_p = log_p + _log_p.sum(1)
          
        return w, log_p, log_det

    def reverse(self, w, conditions):
        output = w.view(-1,16,768)
    
        # Headers
        output_by_levels = output.view(-1,4,4,768).permute(1,0,2,3)
        input_by_levels = []
        for feature_level in range(self.n_levels): # ['layout', 'object', 'attribute', 'color']
            # 2nd Layer
            output_by_level = output_by_levels[feature_level]
            output_by_vectors = output_by_level.view(-1,4,768,1,1).permute(1,0,2,3,4)
            input_by_vectors = []
            for vector_idx in range(self.n_vectors):
                condition = conditions[-1]
                output_by_vector = output_by_vectors[vector_idx]
                header_by_vector = self.headers[feature_level][1][vector_idx]
                input_by_vector = header_by_vector.reverse(output_by_vector, condition, split=None)
                input_by_vectors.append(input_by_vector) # (b,192,2,2)
            input_by_vectors = torch.cat(input_by_vectors, dim=1) # 4x(b,192,2,2) -> (b,4x192,2,2)       
            
            # 1st Layer
            condition = conditions[-2]
            header_by_level = self.headers[feature_level][0]
            input_by_level = header_by_level.reverse(input_by_vectors, condition, split=None)
            input_by_levels.append(input_by_level) # (b,192,4,4)
        input_by_levels = torch.cat(input_by_levels, dim=1) # 4x(b,192,4,4) -> (b,4x192,4,4)
        
        # Blocks
        input = input_by_levels
        for block, condition in zip(self.blocks[::-1], conditions[len(self.blocks)-1::-1]):
            input = block.reverse(input, condition, split=None)
            
        return input

    
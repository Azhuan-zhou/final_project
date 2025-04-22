# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_points
import logging as log
from models.base_modules import PositionalEncoding, Transformer
from models.geo_model import GeoModel
from models.tex_model import TexModel
from SMPL_query import SMPL_query
from models.Feature import FeatureExtractor



class Renderer(torch.nn.Module):
    def __init__(self, cfg,watertight, can_V):
        super().__init__()
        self.cfg = cfg
        self.log_dict = {}
        self._init_log_dict()
        #feature extractor
        self.use_global_feature = cfg.use_global_feature
        self.use_point_level_feature = cfg.use_point_level_feature
        self.use_pixel_align_feature = cfg.use_pixel_align_feature
        self.feature_extractor = FeatureExtractor()
        self.use_trans = cfg.use_trans
        if cfg.use_global_feature:
            if cfg.use_point_level_feature and cfg.use_pixel_align_feature:
                self.conv1d_reprojection = nn.Conv1d(128 + 96 + 96, 32, 1)
            elif cfg.use_point_level_feature and not cfg.use_pixel_align_feature:
                self.conv1d_reprojection = nn.Conv1d(128 + 96, 32, 1)
            elif not cfg.use_point_level_feature and cfg.use_pixel_align_feature:
                self.conv1d_reprojection = nn.Conv1d(128 + 96, 32, 1)
            elif not cfg.use_point_level_feature and not cfg.use_pixel_align_feature:
                self.conv1d_reprojection = nn.Conv1d(128, 32, 1)
        else:
            if cfg.use_point_level_feature and cfg.use_pixel_align_feature:
                self.conv1d_reprojection = nn.Conv1d(96 + 96, 32, 1)
            elif cfg.use_point_level_feature and not cfg.use_pixel_align_feature:
                self.conv1d_reprojection = nn.Conv1d(96, 32, 1)
            elif not cfg.use_point_level_feature and cfg.use_pixel_align_feature:
                self.conv1d_reprojection = nn.Conv1d(96, 32, 1)
            else:
                self.conv1d_reprojection = None
        self.transformer = Transformer(32)

        self.pos_enc = PositionalEncoding(num_freqs=6)
        self.view_enc = PositionalEncoding(num_freqs=4)
     
        
        
        # decoder
        self.smpl_query = SMPL_query(watertight['smpl_F'], can_V)
        self.geo_model = GeoModel(self.cfg, self.smpl_query)
        self.tex_model = TexModel(self.cfg, self.smpl_query)

    

    def forward_3D(self,input_data,pts,geo=True,tex=True):
        feats = []
        smpl_v = input_data['smpl_v']
        vis_class = input_data['vis_class']
        if self.use_global_feature:
            global_feature = self.feature_extractor.global_feature(input_data) # 128
            feats.append(global_feature)

        if self.use_point_level_feature:
            point_level_feature = self.feature_extractor.point_level_F(input_data) # [bs, N_rays*N_samples, 96]
            feats.append(point_level_feature)

        if self.use_pixel_align_feature:
            pixel_align_feature = self.feature_extractor.pixel_align_F(input_data) # torch.Size([b, N, 96])
            feats.append(pixel_align_feature)
            
        if feats == []:
            raise ValueError
        
        combined_feats =  torch.cat(feats,dim=-1) 
        if geo:
            pred_sdf, pred_nrm = self.geo_model(combined_feats,pts,smpl_v,vis_class)
        else: 
            pred_sdf, pred_nrm = None, None
        if tex:
            pred_rgb = self.tex_model(combined_feats,pts,smpl_v,vis_class)
        else:
            pred_rgb = None
        return pred_sdf,pred_nrm,pred_rgb
        

    def backward_3D(self, input_data,pred_sdf,pred_nrm,pred_rgb):
        gts = input_data['d']
        rgb = input_data['rgb']
        nrm = input_data['nrm']
        
        # Compute 3D losses
        reco_loss = torch.abs(pred_sdf - gts).mean()

        rgb_loss = torch.abs(pred_rgb - rgb).mean()

        nrm_loss = torch.abs(1 - F.cosine_similarity(pred_nrm, nrm, dim=-1)).mean()

        loss_3D = reco_loss * self.cfg.lambda_sdf + rgb_loss * self.cfg.lambda_rgb + nrm_loss * self.cfg.lambda_nrm
        return loss_3D,reco_loss, rgb_loss, nrm_loss
    
    
    def forward(self,input_data):
        pts = input_data['xyz']
        pred_sdf,pred_nrm,pred_rgb = self.forward_3D(input_data,pts)
        loss,reco_loss, rgb_loss, nrm_loss = self.backward_3D(input_data,pred_sdf,pred_nrm,pred_rgb)
        self.log_dict['Loss_3D/reco_loss'] += reco_loss.item()
        self.log_dict['Loss_3D/rgb_loss'] += rgb_loss.item()
        self.log_dict['Loss_3D/nrm_loss'] += nrm_loss.item()
        self.log_dict['Loss_3D/total_loss'] += loss.item()
        self.log_dict['total_iter_count'] += 1
        return loss
    
    def _init_log_dict(self):
        """Custom logging dictionary.
        """
        self.log_dict['Loss_3D/reco_loss'] = 0
        self.log_dict['Loss_3D/rgb_loss'] = 0
        self.log_dict['Loss_3D/nrm_loss'] = 0
        self.log_dict['Loss_3D/total_loss'] = 0

        self.log_dict['total_iter_count'] = 0
    
    
    def log(self, step, epoch,lr):
        """Log the training information.
        """
        log_text = 'Train: STEP {} - EPOCH {}/{}'.format(step, epoch, self.cfg.epochs)
        self.log_dict['Loss_3D/total_loss'] /= self.log_dict['total_iter_count'] + 1e-6
        log_text += ' | 3D loss: {:>.3E}'.format(self.log_dict['Loss_3D/total_loss'])
        self.log_dict['Loss_3D/reco_loss'] /= self.log_dict['total_iter_count'] + 1e-6
        log_text += ' | reco loss: {:>.3E}'.format(self.log_dict['Loss_3D/reco_loss'])
        self.log_dict['Loss_3D/rgb_loss'] /= self.log_dict['total_iter_count'] + 1e-6
        log_text += ' | rgb loss: {:>.3E}'.format(self.log_dict['Loss_3D/rgb_loss'])
        self.log_dict['Loss_3D/nrm_loss'] /= self.log_dict['total_iter_count'] + 1e-6
        log_text += ' | nrm loss: {:>.3E}'.format(self.log_dict['Loss_3D/nrm_loss'])
        
        log_text += ' | lr: {:>.3E}'.format(lr)
        log.info(log_text)
        
        
        
        
        
        
        
            
            
        

       
    
    

   

    


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
import logging as log
from models.geo_model import GeoModel
from models.tex_model import TexModel
from models.SMPL_query import SMPL_query
from models.Feature import FeatureExtractor, FeatureExtractor_2v
from models.PointFeat import PointFeat, SMPLX


class ReconModel(torch.nn.Module):
    def __init__(self, cfg,smpl_F, can_V):
        super().__init__()
        self.cfg = cfg
        self.log_dict = {}
        self._init_log_dict()
        #feature extractor
        if cfg.views == 4:
            self.feature_extractor = FeatureExtractor(cfg)
            print("Using 4 views")
        elif cfg.views == 2:
            self.feature_extractor = FeatureExtractor_2v(cfg) 
            print("Using 2 views")
        else:
            raise ValueError("Invalid number of views. Choose 2 or 4.")
        self.use_trans = cfg.use_trans
        # decoder
        smpl_query = SMPL_query(smpl_F, can_V)
        if cfg.dataset == 'THuman':
            smpl_F_point_feat = torch.as_tensor(SMPLX().smplx_faces).long()
        elif cfg.body_type == 'cape':
            smpl_F_point_feat  = torch.as_tensor(SMPLX().smpl_faces).long()
        else:
            raise ValueError("Invalid body type. Choose 'smplx' or 'smpl'.")
        
        self.geo_model = GeoModel(self.cfg, smpl_query, smpl_F_point_feat)
        self.tex_model = TexModel(self.cfg, smpl_query, smpl_F_point_feat)

    

    def forward_3D(self,input_data,pts,geo=True,tex=True):
        smpl_v = input_data['smpl_v']
        vis_class = input_data['vis_class']
        color_feats_map, nrm_feats_map = self.feature_extractor.extract_map(input_data)
        if geo:
            pred_sdf, pred_nrm = self.geo_model(color_feats_map, nrm_feats_map,pts,smpl_v,vis_class)
        else: 
            pred_sdf, pred_nrm = None, None
        if tex:
            pred_rgb = self.tex_model(color_feats_map, nrm_feats_map,pts,smpl_v,vis_class)
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
        if loss_3D.dim() == 0:
            loss_3D = loss_3D.unsqueeze(0)
        return loss_3D,reco_loss, rgb_loss, nrm_loss
    
    
    def forward(self,input_data):
        pts = input_data['pts']
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
        log.info('\n',log_text)
        
        
        
        
        
        
        
            
            
        

       
    
    

   

    


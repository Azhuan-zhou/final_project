"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""

import logging as log
import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_encoding import PositionalEncoding

from .networks.encoder import HGFilter
from .networks.mlps import MLP
from .networks.layers import get_activation_class, get_layer_class
from models.PointFeat import PointFeat, SMPLX


class GeoModel(nn.Module):

    def __init__(self, config, smpl_query,smpl_F):

        super().__init__()

        self.cfg = config
        self.query = smpl_query
        self.smpl_F = smpl_F
        self._init_model()

    def _init_model(self):
        """Initialize model.
        """

        log.info("Initializing neural field...")

        if self.cfg.shape_freq != 0 :
            self.nrm_embedder = PositionalEncoding(self.cfg.shape_freq, self.cfg.shape_freq -1, input_dim=self.cfg.pos_dim)
            self.nrm_embedder = self.nrm_embedder.to(self.device)
            self.embed_dim = self.nrm_embedder.out_dim
        else:
            self.nrm_embedder = None
            self.embed_dim = self.cfg.pos_dim

        self.nrm_input_dim = self.embed_dim + self.cfg.feat_dim * 2

        self.sdf_decoder = MLP(self.nrm_input_dim, 1, activation=get_activation_class(self.cfg.activation),
                                    bias=True, layer=get_layer_class(self.cfg.layer_type), num_layers=self.cfg.num_layers,
                                    hidden_dim=self.cfg.hidden_dim, skip=self.cfg.skip)




    def _get_pos_features(self, pts, smpl_v, vis_class, embedder=None, pos_dim=1):
        with torch.no_grad():
            if pos_dim == 1:
                coord_feats = pts[:, :, 2:3]
            elif pos_dim == 3:
                coord_feats = pts
            elif pos_dim == 5:
                out_coord, sdf, normal, v = self.query.interpolate_vis(pts, smpl_v, vis_class)
                coord_feats = torch.cat([out_coord, sdf, v], dim=-1)
            elif pos_dim == 6:
                out_coord, sdf, normal, v = self.query.interpolate_vis(pts, smpl_v, vis_class)
                coord_feats = torch.cat([out_coord, sdf, v, pts[:, :, 2:3]], dim=-1)
            else:
                out_coord, sdf, normal, v = self.query.interpolate_vis(pts, smpl_v, vis_class)
                coord_feats = torch.cat([out_coord, sdf, normal, v], dim=-1) 

            if embedder is not None:
                coord_feats = embedder(coord_feats)
                
        return coord_feats

    def compute_nrm(self, x, color_feats_map, nrm_feats_map, smpl_v, vis_class, eps=0.001):
        """Compute 3D gradient using finite difference.

        Args:
            x (torch.FloatTensor): Coordinate tensor of shape [B, N, 3]
        """
        shape = x.shape

        eps_x = torch.tensor([eps, 0.0, 0.0], device=x.device)
        eps_y = torch.tensor([0.0, eps, 0.0], device=x.device)
        eps_z = torch.tensor([0.0, 0.0, eps], device=x.device)

        # shape: [B, 4, N, 3] -> [B, 4*N, 3]
        x_new = torch.stack([x + eps_x, x + eps_y, x + eps_z, x], dim=1).reshape(shape[0], -1, shape[-1])
        
        # shape: [B, 4*N, 1] -> [B, 4, N]

        pred = self.compute_sdf(x_new, color_feats_map, nrm_feats_map, smpl_v, vis_class)
        pred = pred.view(shape[0], 4, -1)

        grad_x = (pred[:, 0, ...] - pred[:, 3, ...]) 
        grad_y = (pred[:, 1, ...] - pred[:, 3, ...]) 
        grad_z = (pred[:, 2, ...] - pred[:, 3, ...]) 

        return torch.stack([grad_x, grad_y, grad_z], dim=-1)


    def compute_sdf(self, x, color_feats_map, nrm_feats_map, smpl_v, vis_class):
        
        sdf_feats = self.extract_features(color_feats_map, nrm_feats_map, x, smpl_v)

        coord_feats = self._get_pos_features(x, smpl_v, vis_class, self.nrm_embedder, self.cfg.pos_dim)

        return self.sdf_decoder(torch.cat([sdf_feats, coord_feats], dim=-1))



    def forward(self, color_feats_map, nrm_feats_map, pts, smpl_v, vis_class):
        
        pred_sdf = self.compute_sdf(pts, color_feats_map, nrm_feats_map, smpl_v, vis_class)

        pred_nrm = F.normalize(self.compute_nrm(pts, color_feats_map, nrm_feats_map, smpl_v, vis_class),
                                     p=2, dim=-1, eps=1e-5)


        return pred_sdf, pred_nrm
    
    
    def _query_feature(self, feat, uv):
        """:
        param feat: [B, C, H, W] image features
        :param uv: [B, N,2] uv coordinates in the image plane, range [0, 1]
        :return: [B, C, N] image features at the uv coordinates
        """
        xy = torch.clip(uv, -1, 1).clone()
        xy[..., 1] = -xy[..., 1]
       
        xy = xy.unsqueeze(2) # [B, N, 1, 2]
        samples = torch.nn.functional.grid_sample(feat, xy, align_corners=True)   # [B, C, N, 1]
        return samples.squeeze(-1)  # [B,C,N]
    
    def query_feature(self, feats, pts,smplx_pts,point_feat_extractor):
        F_plane_feat,L_plane_feat,B_plane_feat,R_plane_feat = feats[0],feats[1],feats[2],feats[3]
        F_plane_feat1,F_plane_feat2=F_plane_feat.chunk(2,dim=1)
        L_plane_feat1,L_plane_feat2=L_plane_feat.chunk(2,dim=1)
        B_plane_feat1,B_plane_feat2=B_plane_feat.chunk(2,dim=1)
        R_plane_feat1,R_plane_feat2=R_plane_feat.chunk(2,dim=1)
        xy =  pts[:,:,:2]
        zy = pts[:,:,[2,1]]
        

        smplx_xy = smplx_pts[:,:,:2]
        smplx_zy = smplx_pts[:,:,[2,1]]
        
        F_feat = self._query_feature(F_plane_feat1, xy)  # [B, C,N]
        B_feat = self._query_feature(B_plane_feat1, xy)  # [B, C, N]
        R_feat = self._query_feature(R_plane_feat1, zy)  # [B, C, N]
        L_feat = self._query_feature(L_plane_feat1, zy)  # [B, C, N]
        
        
        three_plane_feat=(B_feat+R_feat+L_feat)/3
        triplane_feat=torch.cat([F_feat,three_plane_feat],dim=1)  # 32+32=64
        
        
        smplx_F_feat = self._query_feature(F_plane_feat2, smplx_xy)  # [B, C, N]
        smplx_B_feat = self._query_feature(B_plane_feat2, smplx_xy)
        smplx_R_feat = self._query_feature(R_plane_feat2, smplx_zy)  # [B, C, N]
        smplx_L_feat = self._query_feature(L_plane_feat2, smplx_zy)  # [B, C, N]
        
        smplx_three_plane_feat=(smplx_B_feat+smplx_R_feat+smplx_L_feat)/3
        smplx_triplane_feat=torch.cat([smplx_F_feat,smplx_three_plane_feat],dim=1)    # 32+32=64
        bary_centric_feat=point_feat_extractor.query_barycentirc_feats(pts,smplx_triplane_feat.permute(0,2,1)) 
        
        feat=torch.cat([triplane_feat,bary_centric_feat.permute(0,2,1)],dim=1)  # 64+64=128
        return feat
    
    def extract_features(self,color_feats_map,nrm_feats_map,pts,smplx_pts):
        batch_size =  smplx_pts.shape[0]
        smpl_F = self.smpl_F.unsqueeze(0).expand(batch_size, -1, -1).to(smplx_pts.device)
        point_feat_extractor = PointFeat(smplx_pts,smpl_F)
        color_feat = self.query_feature(color_feats_map, pts, smplx_pts,point_feat_extractor)
        nrm_feat = self.query_feature(nrm_feats_map, pts, smplx_pts,point_feat_extractor)
        feats = torch.cat([color_feat, nrm_feat], dim=1) # [B, C, N]
        return feats.permute(0,2,1) # [B, N, C]

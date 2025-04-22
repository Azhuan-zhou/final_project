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



class GeoModel(nn.Module):

    def __init__(self, config, smpl_query):

        super().__init__()

        self.cfg = config
        self.query = smpl_query
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

    def compute_nrm(self, x, feats, smpl_v, vis_class, eps=0.001):
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

        pred = self.compute_sdf(x_new, feats, smpl_v, vis_class)
        pred = pred.view(shape[0], 4, -1)

        grad_x = (pred[:, 0, ...] - pred[:, 3, ...]) 
        grad_y = (pred[:, 1, ...] - pred[:, 3, ...]) 
        grad_z = (pred[:, 2, ...] - pred[:, 3, ...]) 

        return torch.stack([grad_x, grad_y, grad_z], dim=-1)


    def compute_sdf(self, x, sdf_feats, smpl_v, vis_class):

        coord_feats = self._get_pos_features(x, smpl_v, vis_class, self.nrm_embedder, self.cfg.pos_dim)

        return self.sdf_decoder(torch.cat([sdf_feats, coord_feats], dim=-1))



    def forward(self, feats, pts, smpl_v, vis_class):

        pred_sdf = self.compute_sdf(pts, feats, smpl_v, vis_class)

        pred_nrm = F.normalize(self.compute_nrm(pts, feats, smpl_v, vis_class),
                                     p=2, dim=-1, eps=1e-5)


        return pred_sdf, pred_nrm

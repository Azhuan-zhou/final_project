"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""

import logging as log
import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding
from .networks.encoder import HGFilter
from .networks.mlps import MLP
from .networks.layers import get_activation_class, get_layer_class



class TexModel(nn.Module):

    def __init__(self, config, smpl_query):

        super().__init__()

        self.cfg = config
        self.query = smpl_query
        self._init_model()

    def _init_model(self):
        """Initialize model.
        """

        log.info("Initializing neural field...")

        if self.cfg.color_freq != 0 :
            self.rgb_embedder = PositionalEncoding(self.cfg.color_freq, self.cfg.color_freq -1, input_dim=self.cfg.pos_dim)
            self.rgb_embedder = self.rgb_embedder.to(self.device)
            self.embed_dim  = self.rgb_embedder.out_dim
        else:
            self.rgb_embedder = None
            self.embed_dim = self.cfg.pos_dim

        self.rgb_input_dim = self.embed_dim + self.cfg.feat_dim * 2

        self.rgb_decoder = MLP(self.rgb_input_dim, 3, activation=get_activation_class(self.cfg.activation),
                                    bias=True, layer=get_layer_class(self.cfg.layer_type), num_layers=self.cfg.num_layers-1,
                                    hidden_dim=self.cfg.hidden_dim // 2)



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


    def compute_rgb(self, x, rgb_feats, smpl_v, vis_class):
        coord_feats = self._get_pos_features(x, smpl_v, vis_class, self.rgb_embedder, self.cfg.pos_dim)

        return self.rgb_decoder(torch.cat([rgb_feats, coord_feats], dim=-1), sigmoid=True)


    def forward(self, feats, pts, smpl_v, vis_class):

        pred_rgb = self.compute_rgb(pts, feats, smpl_v, vis_class)

        return pred_rgb
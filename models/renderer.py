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

from pytorch3d.ops.knn import knn_points
from models.volumetric_rendering.ray_marcher import MipRayMarcher2
from models.volumetric_rendering import math_utils
from models.base_modules import PositionalEncoding, Transformer


from models.Feature import FeatureExtractor



class Renderer(torch.nn.Module):
    def __init__(self, use_global_feature=True, use_point_level_feature=True, use_pixel_align_feature=True, use_trans=False, use_NeRF_decoder=False):
        super().__init__()
        self.use_global_feature = use_global_feature
        self.use_point_level_feature = use_point_level_feature
        self.use_pixel_align_feature = use_pixel_align_feature
        self.feature_extractor = FeatureExtractor()
        self.use_trans = use_trans
        self.use_NeRF_decoder = use_NeRF_decoder
        self.ray_marcher = MipRayMarcher2()
        if use_global_feature:
            if use_point_level_feature and use_pixel_align_feature:
                self.conv1d_reprojection = nn.Conv1d(134 + 96 + 96, 32, 1)
            elif use_point_level_feature and not use_pixel_align_feature:
                self.conv1d_reprojection = nn.Conv1d(134 + 96, 32, 1)
            elif not use_point_level_feature and use_pixel_align_feature:
                self.conv1d_reprojection = nn.Conv1d(134 + 96, 32, 1)
            elif not use_point_level_feature and not use_pixel_align_feature:
                self.conv1d_reprojection = nn.Conv1d(134, 32, 1)
        else:
            if use_point_level_feature and use_pixel_align_feature:
                self.conv1d_reprojection = nn.Conv1d(96 + 96, 32, 1)
            elif use_point_level_feature and not use_pixel_align_feature:
                self.conv1d_reprojection = nn.Conv1d(96, 32, 1)
            elif not use_point_level_feature and use_pixel_align_feature:
                self.conv1d_reprojection = nn.Conv1d(96, 32, 1)
            else:
                self.conv1d_reprojection = None
 
   
            
        
        self.transformer = Transformer(32)

        self.pos_enc = PositionalEncoding(num_freqs=6)
        self.view_enc = PositionalEncoding(num_freqs=4)
        # self.view_enc = PositionalEncoding(num_freqs=5)

    

    def forward(self, decoder, ray_origins, ray_directions, near, far, input_data, rendering_options,test_flag=False):
        depths_coarse = self.sample_stratified(ray_origins, near, far, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape
        if test_flag:
            rendering_options.update({'density_noise': 0})
        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)

        smpl_query_pts, _, coarse_canonical_pts, coarse_canonical_viewdir, pts_mask = self.get_pts(input_data, sample_coordinates, sample_directions)
        if self.use_global_feature:
            global_feature = self.feature_extractor.global_feature(input_data,sample_coordinates) # 134
        else:
            global_feature = torch.zeros((*smpl_query_pts.shape[:2], 134)).to(smpl_query_pts.device)

        if self.use_point_level_feature:
            point_level_feature = self.feature_extractor.point_level_F(input_data, coarse_canonical_pts) # [bs, N_rays*N_samples, 96]

        else:
            point_level_feature = torch.zeros((*smpl_query_pts.shape[:2], 96)).to(smpl_query_pts.device)

        if self.use_pixel_align_feature:
            pixel_align_feature = self.feature_extractor.pixel_align_F(input_data, coarse_canonical_pts) # torch.Size([b, N, 96])
        else:
            pixel_align_feature = torch.zeros((*smpl_query_pts.shape[:2], 96)).to(smpl_query_pts.device)

        out = {}
        chunk = 700000
        for i in range(0, coarse_canonical_pts.shape[1], chunk):
            out_part = self.run_model(global_feature[:, i:i+chunk], point_level_feature[:, i:i+chunk], pixel_align_feature[:, i:i+chunk], decoder, coarse_canonical_pts[:, i:i+chunk], coarse_canonical_viewdir[:, i:i+chunk], rendering_options)
            for k in out_part.keys():
                if k not in out.keys():
                    out[k] = []
                out[k].append(out_part[k]) 
        out = {k : torch.cat(out[k], 1) for k in out.keys()}

        colors_coarse = torch.zeros((*pts_mask.shape, 3), device=pts_mask.device)
        densities_coarse = torch.zeros((*pts_mask.shape, 1), device=pts_mask.device)
        colors_coarse[pts_mask==1] = out['rgb'].squeeze(0)
        densities_coarse[pts_mask==1] = out['sigma'].squeeze(0)
        densities_coarse[pts_mask==0] = -80

        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        if N_importance > 0:
            _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
            smpl_query_pts, _, coarse_canonical_pts, _, pts_mask = self.get_pts(input_data, sample_coordinates, sample_directions)
            if self.use_global_feature:
                global_feature = self.feature_extractor.global_feature(input_data,sample_coordinates)
            else:
                global_feature = torch.zeros((*smpl_query_pts.shape[:2], 134)).to(smpl_query_pts.device)
            if self.use_point_level_feature:
                point_level_feature = self.feature_extractor.point_level_F(input_data, coarse_canonical_pts) # [bs, N_rays*N_samples, 96]
            else:
                point_level_feature = torch.zeros((*smpl_query_pts.shape[:2], 96)).to(smpl_query_pts.device)

            if self.use_pixel_align_feature:
                pixel_align_feature = self.feature_extractor.pixel_align_F(input_data, coarse_canonical_pts) # torch.Size([b, N, 96])
            else:
                pixel_align_feature = torch.zeros((*smpl_query_pts.shape[:2], 96)).to(smpl_query_pts.device)
            
            
            out = self.run_model(global_feature, point_level_feature,pixel_align_feature, decoder, sample_coordinates, sample_directions, rendering_options)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)

            all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                  depths_fine, colors_fine, densities_fine)

            # Aggregate
            rgb_final, depth_final, weights = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
        else:
            rgb_final, depth_final, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, ray_directions, rendering_options)


        return rgb_final, depth_final, weights.sum(2)

    
    def combine_features(self, global_feature, point_level_feature, pixel_align_feature):
        """
        point_1d_feature: [bs, N_rays*N_samples, 134]
        point_2d_feature: [bs, N_rays*N_samples, 96]
        point_3d_feature: [bs, N_rays*N_samples, 96]
        """
        if self.use_global_feature:
            if self.use_point_level_feature and self.use_pixel_align_feature:
                combined_feature = torch.cat([global_feature, point_level_feature,pixel_align_feature],dim=-1)
            elif self.use_point_level_feature and not self.use_pixel_align_feature:
                combined_feature = torch.cat([global_feature, point_level_feature],dim=-1)
            elif not self.use_point_level_feature and self.use_pixel_align_feature:
                combined_feature = torch.cat([global_feature, pixel_align_feature],dim=-1)
            elif not self.use_point_level_feature and not self.use_pixel_align_feature:
                combined_feature = global_feature
        else:
            if self.use_point_level_feature and self.use_pixel_align_feature:
                combined_feature = torch.cat([point_level_feature,pixel_align_feature],dim=-1)
            elif self.use_point_level_feature and not self.use_pixel_align_feature:
                combined_feature = point_level_feature
            elif not self.use_point_level_feature and self.use_pixel_align_feature:
                combined_feature = pixel_align_feature
            else:
                combined_feature = torch.zeros((*global_feature.shape[:2], 32)).to(global_feature.device)
        return  combined_feature
    def run_model(self, global_feature, point_level_feature, pixel_align_feature, decoder, sample_coordinates, sample_directions, options):
        sampled_features = global_feature
        # if point_2d_feature is not None and point_3d_feature is not None:
        combined_feature = self.combine_features(global_feature, point_level_feature, pixel_align_feature)
        if self.conv1d_reprojection is not None:
            sampled_features = self.conv1d_reprojection(combined_feature.permute(0,2,1)).permute(0,2,1)
        else:
            sampled_features = combined_feature
        if self.use_trans:
            
            sampled_features = self.transformer(sampled_features.permute(0,2,1,3).reshape(-1, sampled_features.shape[1], sampled_features.shape[-1])).permute(1,0,2).reshape(-1, 3, *sampled_features.shape[2:])

        if not self.use_NeRF_decoder:
            out = decoder(sampled_features, sample_directions)
        else:
            out = decoder(self.pos_enc(sample_coordinates.squeeze(0)), sampled_features.squeeze(0), self.view_enc(sample_directions.squeeze(0)))

        # out = decoder(sampled_features.permute(0,2,1,3)[pts_mask==1].permute(1,0,2).unsqueeze(0), 2sample_directions[pts_mask==1].unsqueeze(0))
        if options.get('density_noise', 0) > 0:
            out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']
        return out
    
    def get_pts(self,input_data,sample_coordinates,sample_directions):
        
        R = input_data['params']['R'] # [bs, 3, 3]
        Th = input_data['params']['Th'] #.astype(np.float32) [bs, 1, 3]
        smpl_query_pts = torch.matmul(sample_coordinates - Th, R) # [bs, N_rays*N_samples, 3]
        smpl_query_viewdir = torch.matmul(sample_directions, R)

        # human sample
        tar_smpl_pts = input_data['vertices'] # [bs, 6890, 3]
        tar_smpl_pts = torch.matmul(tar_smpl_pts - Th, R) # [bs, 6890, 3]
        distance, vertex_id, _ = knn_points(smpl_query_pts.float(), tar_smpl_pts.float(), K=1)
        distance = distance.view(distance.shape[0], -1)
        pts_mask = torch.zeros_like(smpl_query_pts[...,0]).int()
        threshold = 0.05 ** 2 
        pts_mask[distance < threshold] = 1
        smpl_query_pts = smpl_query_pts[pts_mask==1].unsqueeze(0)
        smpl_query_viewdir = smpl_query_viewdir[pts_mask==1].unsqueeze(0)   

        coarse_canonical_pts, coarse_canonical_viewdir = self.coarse_deform_target2c(input_data['params'], input_data['vertices'], input_data['t_params'], smpl_query_pts, smpl_query_viewdir)
        return smpl_query_pts, smpl_query_viewdir, coarse_canonical_pts, coarse_canonical_viewdir, pts_mask

    def sort_samples(self, all_depths, all_colors, all_densities):
        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        return all_depths, all_colors, all_densities

    def unify_samples(self, depths1, colors1, densities1, depths2, colors2, densities2):
        all_depths = torch.cat([depths1, depths2], dim = -2)
        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_densities = torch.cat([densities1, densities2], dim = -2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))

        return all_depths, all_colors, all_densities

    def sample_stratified(self, ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(0,
                                    1,
                                    depth_resolution,
                                    device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
            depth_delta = 1/(depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = math_utils.linspace(ray_start, ray_end, depth_resolution).permute(1,2,0,3)
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                # depths_coarse += torch.rand_like(depths_coarse) * depth_delta[..., None]
            else:
                depths_coarse = torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
                depth_delta = (ray_end - ray_start)/(depth_resolution - 1)
                # depths_coarse += torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse

    def sample_importance(self, z_vals, weights, N_importance):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1) # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = torch.nn.functional.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = torch.nn.functional.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1],
                                             N_importance).detach().reshape(batch_size, num_rays, N_importance, 1)
        return importance_z_vals

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                                   # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds-1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[...,1]-cdf_g[...,0]
        denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                             # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
        return samples

   

    

   

    


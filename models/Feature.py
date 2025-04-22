
from ipaddress import _IPAddressBase
import math
import os
import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import pickle
import numpy as np
from pytorch3d.ops.knn import knn_points
import spconv.pytorch as spconv
from models.base_modules import PositionalEncoding, SparseConvNet, Transformer, ResNet18Classifier
from models.utils import SMPL_to_tensor, read_pickle
from models.Transformer import ViTVQ
from models.HGFilters import HGFilter
from models.PointFeat import PointFeat
from NormalNet  import NormalNet


def feat_select(feat, select):

    # feat [B, featx2, N]
    # select [B, 1, N]
    # return [B, feat, N]

    dim = feat.shape[1] // 2
    idx = torch.tile((1-select), (1, dim, 1))*dim + \
        torch.arange(0, dim).unsqueeze(0).unsqueeze(2).type_as(select)
    feat_select = torch.gather(feat, 1, idx.long())

    return feat_select

class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # extract features from images
        self.image_filter=ViTVQ(image_size=512,channels=9) # 1d
        self.F_filter = HGFilter(num_modules=2, in_dim=3)
        

        self.encoder_2d_feature = ResNet18Classifier() # 2d 3d
        
        self.conv1d_projection_1 = nn.Conv1d(96, 32, 1)
        
        self.encoder_3d = SparseConvNet(num_layers=4)
        self.conv1d_projection_2 = nn.Conv1d(192, 96, 1)
        self.rgb_enc = PositionalEncoding(num_freqs=5)

        # load SMPL model
        neutral_smpl_path = os.path.join('data/smplx', 'SMPXL_MALE.pkl')
        self.SMPL_NEUTRAL = SMPL_to_tensor(read_pickle(neutral_smpl_path), device=torch.device('cuda', torch.cuda.current_device()))
        
        
    def get_grid_coords(self, pts, sp_input):
        # convert xyz to the voxel coordinate dhw
        dhw = pts[..., [2, 1, 0]]
        min_dhw = sp_input['bounds'][:, 0, [2, 1, 0]]
        dhw = dhw - min_dhw[:, None]
        dhw = dhw / torch.tensor([0.005, 0.005, 0.005]).to(dhw)
        # convert the voxel coordinate to [-1, 1]
        out_sh = torch.tensor(sp_input['out_sh']).to(dhw)
        dhw = dhw / out_sh * 2 - 1
        # convert dhw to whd, since the occupancy is indexed by dhw
        grid_coords = dhw[..., [2, 1, 0]]
        return grid_coords
    
    def get_normal(self, in_tensor_dict):
        with torch.no_grad():
            feat_lst = []
            nmlF, nmlB = self.normal_filter(in_tensor_dict)
            feat_lst.append(nmlF)  # [1, 3, 512, 512]
            feat_lst.append(nmlB)  # [1, 3, 512, 512]
            in_filter = torch.cat(feat_lst, dim=1)

        return in_filter
    
    def global_feature(self, input_data):
        """

        """
        
        front_view_img = input_data['front_image'] # [bs, 3, 512, 512]
        left_view_img = input_data['side_images'][2] # [bs, 3, 512, 512]
        back_view_img = input_data['side_images'][3]# [bs, 3, 512, 512]
        right_view_img = input_data['side_images'][4] # [bs, 3, 512, 512]
        normal_F = input_data['front_nrm_img'] # [bs, 3, 512, 512]
        normal_B = input_data['normal_B'] # [bs, 3, 512, 512]
        fuse_image_F=torch.cat([front_view_img,normal_F,normal_B], dim=1)

        multi_views={
            "image_L":left_view_img,
            "image_B":back_view_img,
            "image_R":right_view_img,
            
        }
        F_plane_feat,L_plane_feat,B_plane_feat,R_plane_feat = self.image_filter(fuse_image_F, multi_views) # [bs, 32, 128, 128]) * 4
        features_F = self.F_filter(normal_F)  # [(B,hg_dim,128,128) * 4]
        features_B = self.F_filter(normal_B)  # [(B,hg_dim,128,128) * 4]
        
        F_plane_feat1,F_plane_feat2=F_plane_feat.chunk(2,dim=1)
        L_plane_feat1,L_plane_feat2=L_plane_feat.chunk(2,dim=1)
        B_plane_feat1,B_plane_feat2=B_plane_feat.chunk(2,dim=1)
        R_plane_feat1,R_plane_feat2=R_plane_feat.chunk(2,dim=1)
        
        in_feat=torch.cat([features_F[-1],features_B[-1]], dim=1)
        
        
        pts = input_data['xyz'] # [bs,N, 3]
        xy =  pts[:,:2]
        zy = pts[:,[2,1]]
       
        smpl_pts = input_data['smpl_v'] 
        
        if smpl_pts.shape[0] == 1:
            smpl_pts = smpl_pts.unsqueeze(0)
        smpl_xy = smpl_pts[:,:2]
        smpl_zy = smpl_pts[:,[2,1]]
        
       
        self.point_feat_extractor = PointFeat(input_data['smpl_v'],
                                               self.SMPL_NEUTRAL['f'])
        
        #vis = input_data['vis']
        #vis = self.point_feat_extractor.query(pts, smpl_vis)
        
        F_feat = self._query_feature(F_plane_feat1, xy)  # [B, N, C]
        B_feat = self._query_feature(B_plane_feat1, xy)  # [B, N, C]
        R_feat = self._query_feature(R_plane_feat1, zy)  # [B, N, C]
        L_feat = self._query_feature(L_plane_feat1, zy)  # [B, N, C]
        
        in_feat_xy = self._query_feature(in_feat,xy) # [B,  N, C]
        #normal_feat=feat_select(in_feat_xy,vis)
        
        
        three_plane_feat=(B_feat+R_feat+L_feat)/3
        triplane_feat=torch.cat([F_feat,three_plane_feat],dim=1)  # 32+32=64
        
        
        smpl_F_feat = self._query_feature(F_plane_feat2, smpl_xy)  # [B, N, C]
        smpl_B_feat = self._query_feature(B_plane_feat2, smpl_xy)
        smpl_R_feat = self._query_feature(R_plane_feat2, smpl_zy)  # [B, N, C]
        smpl_L_feat = self._query_feature(L_plane_feat2, smpl_zy)  # [B, N, C]
        
        smpl_three_plane_feat=(smpl_B_feat+smpl_R_feat+smpl_L_feat)/3
        smpl_triplane_feat=torch.cat([smpl_F_feat,smpl_three_plane_feat],dim=1)    # 32+32=64
        bary_centric_feat=self.point_feat_extractor.query_barycentirc_feats(pts,smpl_triplane_feat) 
        
        global_feat=torch.cat([triplane_feat,bary_centric_feat.permute(0,2,1)],dim=1)  # 64+64=128
        return global_feat
        
        
    def point_level_F(self, input_data,):
        # get vertex feature to form sparse convolution tensor
        obs_input_img = input_data['front_image'] # [bs, 3, 512, 512]
        obs_input_feature = self.encoder_2d_feature(obs_input_img, extract_feature=True) # [bs, 64, 256, 256]
        
       
        
        bs = obs_input_img.shape[0]
        obs_vertex_pts = input_data['smpl_v'] # [bs, 6890, 3]
        obs_smpl_vertex_mask = input_data['vis_class']
        obs_uv = obs_vertex_pts[:,:,:2] # [bs, 6890, 2]
        obs_vertex_feature = self._query_feature(obs_input_feature, obs_uv)  # [B, N, C]
        # obs_img = obs_input_img.reshape(-1, *obs_input_img.shape[2:])
        obs_vertex_rgb = self._query_feature(obs_input_img, obs_uv)  # [B, N, C]
        
        #   obs_vertex_rgb = obs_vertex_rgb.view(bs, -1 , *obs_vertex_rgb.shape[1:]).transpose(2,3)
        sh = obs_vertex_rgb.shape
        obs_vertex_rgb = self.rgb_enc(obs_vertex_rgb.reshape(-1,3)).reshape(*sh[:2], 33)[..., :32] # [bs, N_rays*N_samples, 32] 
        obs_vertex_3d_feature = torch.cat((obs_vertex_feature, obs_vertex_rgb), dim=-1) # [bs, N_rays*N_samples, 96] 
        obs_vertex_3d_feature = self.conv1d_projection_1(obs_vertex_3d_feature.permute(0,2,1)).permute(0,2,1)
        obs_vertex_3d_feature[obs_smpl_vertex_mask==0] = 0
        ## vertex points in SMPL coordinates
        smpl_obs_pts = obs_vertex_pts.reshape(bs, -1, 3)
        ## coarse deform target to caonical
        coarse_obs_vertex_canonical_pts = self.coarse_deform_target2c(input_data['obs_params'], obs_vertex_pts, input_data['t_params'], smpl_obs_pts) # [bs, N_rays*N_rand, 3]       
        # prepare sp input
        obs_sp_input, _ = self.prepare_sp_input(input_data['t_vertices'], coarse_obs_vertex_canonical_pts)
        canonical_sp_conv_volume = spconv.core.SparseConvTensor(obs_vertex_3d_feature.reshape(-1, obs_vertex_3d_feature.shape[-1]), obs_sp_input['coord'], obs_sp_input['out_sh'], obs_sp_input['batch_size']) # [bs, 32, 96, 320, 384] z, y, x
        
        smpl_query_pts = input_data['xyz'] # [bs, N, 3]
        distance, vertex_id, _ = knn_points(smpl_query_pts.float(),  obs_vertex_pts.float(), K=1)
        distance = distance.view(distance.shape[0], -1)
        pts_mask = torch.zeros_like(smpl_query_pts[...,0]).int()
        threshold = 0.05 ** 2 
        pts_mask[distance < threshold] = 1
        smpl_query_pts = smpl_query_pts[pts_mask==1].unsqueeze(0)
        coarse_canonical_pts = self.coarse_deform_target2c(input_data['obs_params'], obs_vertex_pts, input_data['t_params'], smpl_query_pts)
        
        
        grid_coords = self.get_grid_coords(coarse_canonical_pts, obs_sp_input)
        # grid_coords = grid_coords.view(bs, -1, 3)
        grid_coords = grid_coords[:, None, None]
        point_level_feature = self.encoder_3d(canonical_sp_conv_volume, grid_coords) # torch.Size([b, 390, 1024*64])
        point_level_feature = self.conv1d_projection_2(point_level_feature.permute(0,2,1)).permute(0,2,1) # torch.Size([b, N, 96])
        return  point_level_feature
    
           
        
    def pixel_align_F(self, input_data):
        obs_input_img = input_data['front_image']  # [bs, 3, 512, 512]
        obs_input_feature = self.encoder_2d_feature(obs_input_img, extract_feature=True) # [bs, 64, 256, 256]
         # extract pixel aligned 2d feature
        bs = obs_input_img.shape[0]
        
        src_uv = input_data['xyz'][:,:2] # [bs, 3]
        point_pixel_feature = self._query_feature(obs_input_feature, src_uv)  # [B, N, C]
       
        # extract pixel aligned rgb feature
        point_pixel_rgb = self._query_feature(obs_input_img, src_uv)  # [B, N, C]
     
        sh = point_pixel_rgb.shape
        point_pixel_rgb = self.rgb_enc(point_pixel_rgb.reshape(-1,3)).reshape(*sh[:2], 33)[..., :32] # [bs, N_rays*N_samples, 32] 
        pixel_level_feature = torch.cat((point_pixel_feature, point_pixel_rgb), dim=-1) # [bs, N_rays*N_samples, 96] 
        return pixel_level_feature
            
    def _query_feature(self, feat, uv):
        '''
        extract image features at floating coordinates with bilinear interpolation
        args:
            feat: [B, C, H, W] image features
            uv: [B, N, 2] normalized image coordinates ranged in [-1, 1]
        return:
            [B, N, C] sampled pixel values
        '''
        xy = torch.clip(uv, -1, 1).clone()
        xy[..., 1] = -xy[..., 1]

        samples = torch.nn.functional.grid_sample(feat, xy.unsqueeze(2), align_corners=True)
        return samples[:, :, :, 0].permute(0, 2, 1)
            
    
        
    def coarse_deform_target2c(self, params, vertices, t_params, query_pts, query_viewdirs = None):
        bs = query_pts.shape[0]
        # joints transformation
        A, joints = get_transform_params_torch(self.SMPL_NEUTRAL, params)
        smpl_pts = vertices 
        _, vert_ids, _ = knn_points(query_pts.float(), smpl_pts.float(), K=1)
        bweights = self.SMPL_NEUTRAL['weights'][vert_ids].view(*vert_ids.shape[:2], 24)#.to(vertices.device)
        # From smpl space target pose to smpl space canonical pose
        A = torch.matmul(bweights, A.reshape(bs, 24, -1))
        A = torch.reshape(A, (bs, -1, 4, 4))
        can_pts = query_pts - A[..., :3, 3]
        R_inv = torch.inverse(A[..., :3, :3].float())
        can_pts = torch.matmul(R_inv, can_pts[..., None]).squeeze(-1)
        if query_viewdirs is not None:
            query_viewdirs = torch.matmul(R_inv, query_viewdirs[..., None]).squeeze(-1)
        self.mean_shape = True
        if self.mean_shape:
            posedirs = self.SMPL_NEUTRAL['posedirs'] #.to(vertices.device).float()
            pose_ = params['poses']
            ident = torch.eye(3).to(pose_.device).float()
            batch_size = pose_.shape[0] #1
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
            pose_offsets = torch.matmul(pose_feature.unsqueeze(1), posedirs.view(6890*3, -1).transpose(1,0).unsqueeze(0)).view(batch_size, -1, 3)
        
            pose_offsets = torch.gather(pose_offsets, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            can_pts = can_pts - pose_offsets
            # To mean shape
            shapedirs = self.SMPL_NEUTRAL['shapedirs']  #.to(pose_.device)
            shape_offset = torch.matmul(shapedirs.unsqueeze(0), torch.reshape(params['shapes'], (batch_size, 1, 10, 1))).squeeze(-1)
            shape_offset = torch.gather(shape_offset, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            can_pts = can_pts - shape_offset
        # From T To Big Pose        
        big_pose_params = t_params #self.big_pose_params(params)
        if self.mean_shape:
            pose_ = big_pose_params['poses']
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
            pose_offsets = torch.matmul(pose_feature.unsqueeze(1), posedirs.view(6890*3, -1).transpose(1,0).unsqueeze(0)).view(batch_size, -1, 3)
            pose_offsets = torch.gather(pose_offsets, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            can_pts = can_pts + pose_offsets
            # To mean shape
            # shape_offset = torch.matmul(shapedirs.unsqueeze(0), torch.reshape(big_pose_params['shapes'].cuda(), (batch_size, 1, 10, 1))).squeeze(-1)
            # shape_offset = torch.gather(shape_offset, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            # can_pts = can_pts + shape_offset
        A, joints = get_transform_params_torch(self.SMPL_NEUTRAL, big_pose_params)
        A = torch.matmul(bweights, A.reshape(bs, 24, -1))
        A = torch.reshape(A, (bs, -1, 4, 4))
        can_pts = torch.matmul(A[..., :3, :3], can_pts[..., None]).squeeze(-1)
        can_pts = can_pts + A[..., :3, 3]
        if query_viewdirs is not None:
            query_viewdirs = torch.matmul(A[..., :3, :3], query_viewdirs[..., None]).squeeze(-1)
            return can_pts, query_viewdirs
        return can_pts
        
def get_transform_params_torch(smpl, params):
    """ obtain the transformation parameters for linear blend skinning
    """
    v_template = smpl['v_template']

    # add shape blend shapes
    shapedirs = smpl['shapedirs']
    betas = params['shapes']

    v_shaped = v_template[None] + torch.sum(shapedirs[None] * betas[:,None], dim=-1).float()

    # add pose blend shapes
    poses = params['poses'].reshape(-1, 3)
    # bs x 24 x 3 x 3
    rot_mats = batch_rodrigues_torch(poses).view(params['poses'].shape[0], -1, 3, 3)

    # obtain the joints
    joints = torch.matmul(smpl['J_regressor'][None], v_shaped) # [bs, 24 ,3]

    # obtain the rigid transformation
    parents = smpl['kintree_table'][0]

    A = get_rigid_transformation_torch(rot_mats, joints, parents)

  

    return A, joints


def batch_rodrigues_torch(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = torch.norm(poses + 1e-8, p=2, dim=1, keepdim=True)
    rot_dir = poses / angle

    cos = torch.cos(angle)[:, None]
    sin = torch.sin(angle)[:, None]

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), device=poses.device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1)
    K = K.reshape([batch_size, 3, 3])

    ident = torch.eye(3)[None].to(poses.device)
    rot_mat = ident + sin * K + (1 - cos) * torch.matmul(K, K)

    return rot_mat


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def get_rigid_transformation_torch(rot_mats, joints, parents):
    """
    rot_mats: bs x 24 x 3 x 3
    joints: bs x 24 x 3
    parents: 24
    """
    # obtain the relative joints
    bs = joints.shape[0]
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    # create the transformation matrix
    transforms_mat = torch.cat([rot_mats, rel_joints[..., None]], dim=-1)
    padding = torch.zeros([bs, 24, 1, 4], device=rot_mats.device)  #.to(rot_mats.device)
    padding[..., 3] = 1
    transforms_mat = torch.cat([transforms_mat, padding], dim=-2)

    # rotate each part
    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)
    transforms = torch.stack(transform_chain, dim=1)

    # obtain the rigid transformation
    padding = torch.zeros([bs, 24, 1], device=rot_mats.device)  #.to(rot_mats.device)
    joints_homogen = torch.cat([joints, padding], dim=-1)
    rel_joints = torch.sum(transforms * joints_homogen[:, :, None], dim=3)
    transforms[..., 3] = transforms[..., 3] - rel_joints

    return transforms

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = torch.sqrt(arr[..., 0] ** 2 + arr[..., 1] ** 2 + arr[..., 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps 
    arr[..., 0] /= lens
    arr[..., 1] /= lens
    arr[..., 2] /= lens
    return arr 

def compute_normal(vertices, faces):
    # norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    norm = torch.zeros(vertices.shape, dtype=vertices.dtype).cuda()
    tris = vertices[:, faces] # [bs, 13776, 3, 3]
    # n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n = torch.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    n = torch.cross(tris[:, :, 1] - tris[:, :, 0], tris[:, :, 2] - tris[:, :, 0]) 
    normalize_v3(n)
    norm[:, faces[:, 0]] += n
    norm[:, faces[:, 1]] += n
    norm[:, faces[:, 2]] += n
    normalize_v3(norm)

    return norm
        
        
        

    
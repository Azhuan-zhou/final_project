
import torch
from models.Transformer import ViTVQ
import pickle
from models.PointFeat import PointFeat
WATERTIGHT_TEMPLATE = 'data/smplx_watertight.pkl'
SMPL_NATURAL = 'data/body_models/smpl/smplx_natural.pkl'

class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # extract features from images
        self.image_color_filter=ViTVQ(image_size=512,channels=9) # 1d
        self.image_nrm_filter=ViTVQ(image_size=512,channels=9) # 1d
    
    def color_feature(self, input_data,pts,smplx_pts):
        front_view_img = input_data['front_image'] # [bs, 3, 512, 512]
        left_view_img = input_data['side_images'][1] # [bs, 3, 512, 512]
        back_view_img = input_data['side_images'][2]# [bs, 3, 512, 512]
        right_view_img = input_data['side_images'][3] # [bs, 3, 512, 512]
        normal_F = input_data['front_nrm_img'] # [bs, 3, 512, 512]
        normal_B = input_data['back_nrm_img'] # [bs, 3, 512, 512]
        fuse_image_F=torch.cat([front_view_img,normal_F,normal_B], dim=1)

        multi_views={
            "image_L":left_view_img,
            "image_B":back_view_img,
            "image_R":right_view_img,
            
        }
        F_plane_feat,L_plane_feat,B_plane_feat,R_plane_feat = self.image_filter(fuse_image_F, multi_views) # [bs, 32, 128, 128]) * 4
        
        F_plane_feat1,F_plane_feat2=F_plane_feat.chunk(2,dim=1)
        L_plane_feat1,L_plane_feat2=L_plane_feat.chunk(2,dim=1)
        B_plane_feat1,B_plane_feat2=B_plane_feat.chunk(2,dim=1)
        R_plane_feat1,R_plane_feat2=R_plane_feat.chunk(2,dim=1)
        xy =  pts[:,:2]
        zy = pts[:,[2,1]]
        
        if smplx_pts.shape[0] == 1:
            smplx_pts = smplx_pts.unsqueeze(0)
        smplx_xy = smplx_pts[:,:2]
        smplx_zy = smplx_pts[:,[2,1]]
        
        F_feat = self._query_feature(F_plane_feat1, xy)  # [B, N, C]
        B_feat = self._query_feature(B_plane_feat1, xy)  # [B, N, C]
        R_feat = self._query_feature(R_plane_feat1, zy)  # [B, N, C]
        L_feat = self._query_feature(L_plane_feat1, zy)  # [B, N, C]
        
        
        three_plane_feat=(B_feat+R_feat+L_feat)/3
        triplane_feat=torch.cat([F_feat,three_plane_feat],dim=1)  # 32+32=64
        
        
        smplx_F_feat = self._query_feature(F_plane_feat2, smplx_xy)  # [B, N, C]
        smplx_B_feat = self._query_feature(B_plane_feat2, smplx_xy)
        smplx_R_feat = self._query_feature(R_plane_feat2, smplx_zy)  # [B, N, C]
        smplx_L_feat = self._query_feature(L_plane_feat2, smplx_zy)  # [B, N, C]
        
        smplx_three_plane_feat=(smplx_B_feat+smplx_R_feat+smplx_L_feat)/3
        smplx_triplane_feat=torch.cat([smplx_F_feat,smplx_three_plane_feat],dim=1)    # 32+32=64
        bary_centric_feat=self.point_feat_extractor.query_barycentirc_feats(pts,smplx_triplane_feat) 
        
        color_feat=torch.cat([triplane_feat,bary_centric_feat.permute(0,2,1)],dim=1)  # 64+64=128
        return color_feat
    
    
    def nrm_feature(self,input_data,pts,smplx_pts):
        front_view_img = input_data['front_image'] # [bs, 3, 512, 512]
        normal_F = input_data['front_nrm_img'] # [bs, 3, 512, 512]
        normal_B = input_data['back_nrm_img'] # [bs, 3, 512, 512]
        fuse_image_F=torch.cat([front_view_img,normal_F,normal_B], dim=1)
        
        left_view_img = input_data['side_normals'][1] # [bs, 3, 512, 512]
        right_view_img = input_data['side_normals'][2]# [bs, 3, 512, 512]

        multi_views={
            "image_L":left_view_img,
            "image_B":normal_B,
            "image_R":right_view_img,
            
        }
        F_plane_feat,L_plane_feat,B_plane_feat,R_plane_feat = self.image_filter(fuse_image_F, multi_views) # [bs, 32, 128, 128]) * 4
        
        
        F_plane_feat1,F_plane_feat2=F_plane_feat.chunk(2,dim=1)
        L_plane_feat1,L_plane_feat2=L_plane_feat.chunk(2,dim=1)
        B_plane_feat1,B_plane_feat2=B_plane_feat.chunk(2,dim=1)
        R_plane_feat1,R_plane_feat2=R_plane_feat.chunk(2,dim=1)
       
        
        xy =  pts[:,:2]
        zy = pts[:,[2,1]]
       
        
        if smplx_pts.shape[0] == 1:
            smplx_pts = smplx_pts.unsqueeze(0)
        smplx_xy = smplx_pts[:,:2]
        smplx_zy = smplx_pts[:,[2,1]]
        
        F_feat = self._query_feature(F_plane_feat1, xy)  # [B, N, C]
        B_feat = self._query_feature(B_plane_feat1, xy)  # [B, N, C]
        R_feat = self._query_feature(R_plane_feat1, zy)  # [B, N, C]
        L_feat = self._query_feature(L_plane_feat1, zy)  # [B, N, C]
        
        
        
        three_plane_feat=(B_feat+R_feat+L_feat)/3
        triplane_feat=torch.cat([F_feat,three_plane_feat],dim=1)  # 32+32=64
        
        
        smplx_F_feat = self._query_feature(F_plane_feat2, smplx_xy)  # [B, N, C]
        smplx_B_feat = self._query_feature(B_plane_feat2, smplx_xy)
        smplx_R_feat = self._query_feature(R_plane_feat2, smplx_zy)  # [B, N, C]
        smplx_L_feat = self._query_feature(L_plane_feat2, smplx_zy)  # [B, N, C]
        
        smplx_three_plane_feat=(smplx_B_feat+smplx_R_feat+smplx_L_feat)/3
        smplx_triplane_feat=torch.cat([smplx_F_feat,smplx_three_plane_feat],dim=1)    # 32+32=64
        
        bary_centric_feat=self.point_feat_extractor.query_barycentirc_feats(pts,smplx_triplane_feat) 
        
        nrm_feat=torch.cat([triplane_feat,bary_centric_feat.permute(0,2,1)],dim=1)  # 64+64=128
        return nrm_feat
    
    
    def extract_features(self,input_data,pts,smplx_pts,body_type='smplx'):
        if body_type == 'smplx':
            with open(WATERTIGHT_TEMPLATE, 'rb') as f:
                smpl_F = torch.from_numpy(pickle.load(f)['smpl_F']).float().to(pts.device)
        elif body_type == 'smpl':
            with open(SMPL_NATURAL, 'rb') as f:
                smpl_F = torch.from_numpy(pickle.load(f)['smpl_F']).float().to(pts.device)
        else:
            raise ValueError("Invalid body type. Choose 'smplx' or 'smpl'.")
        self.point_feat_extractor = PointFeat(smplx_pts,smpl_F)
        color_feat = self.color_feature(input_data,pts,smplx_pts)
        nrm_feat = self.nrm_feature(input_data,pts,smplx_pts)
        feats = torch.cat([color_feat, nrm_feat], dim=1)
        return feats
        
        


        
        

    
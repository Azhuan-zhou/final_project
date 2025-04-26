
import torch
from models.Transformer import ViTVQ, ViTVQ_2v
import pickle



class FeatureExtractor(torch.nn.Module):
    def __init__(self,cfg):
        super().__init__()
        
        # extract features from images
        self.image_filter=ViTVQ(image_size=cfg.img_size,channels=9) # 1d
        
    
    def color_feature(self, input_data):
        front_view_img = input_data['front_image'] # [bs, 3, 512, 512]
        left_view_img = input_data['side_images'][0] # [bs, 3, 512, 512]
        back_view_img = input_data['side_images'][1]# [bs, 3, 512, 512]
        right_view_img = input_data['side_images'][2] # [bs, 3, 512, 512]
        normal_F = input_data['front_nrm_img'] # [bs, 3, 512, 512]
        normal_B = input_data['back_nrm_img'] # [bs, 3, 512, 512]
        fuse_image_F=torch.cat([front_view_img,normal_F,normal_B], dim=1)

        multi_views={
            "image_L":left_view_img,
            "image_B":back_view_img,
            "image_R":right_view_img,
            
        }
        F_plane_feat,L_plane_feat,B_plane_feat,R_plane_feat = self.image_filter(fuse_image_F, multi_views) # [bs, 32, 128, 128]) * 4
        
        color_feats = [ F_plane_feat,L_plane_feat,B_plane_feat,R_plane_feat]
        return color_feats
    
    
    
    def nrm_feature(self,input_data):
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
        nrm_feats = [F_plane_feat,L_plane_feat,B_plane_feat,R_plane_feat]
        return  nrm_feats
    
    def extract_map(self, input_data):
       

        color_feats_map = self.color_feature(input_data)
        nrm_feats_map = self.nrm_feature(input_data)
        return color_feats_map, nrm_feats_map
    
class FeatureExtractor_2v(torch.nn.Module):
    def __init__(self,cfg):
        super().__init__()
        
        # extract features from images
        self.image_filter=ViTVQ_2v(image_size=cfg.img_size,channels=9) # 1d
        
    
    def color_feature(self, input_data):
        front_view_img = input_data['front_image'] # [bs, 3, 512, 512]
        back_view_img = input_data['side_images'][1]# [bs, 3, 512, 512]
        normal_F = input_data['front_nrm_img'] # [bs, 3, 512, 512]
        normal_B = input_data['back_nrm_img'] # [bs, 3, 512, 512]
        fuse_image_F=torch.cat([front_view_img,normal_F,normal_B], dim=1)

        multi_views={
            "image_B":back_view_img,
            
        }
        F_plane_feat,B_plane_feat = self.image_filter(fuse_image_F, multi_views) # [bs, 32, 128, 128]) * 4
        
        color_feats = [ F_plane_feat,B_plane_feat]
        return color_feats
    
    
    
    def nrm_feature(self,input_data):
        front_view_img = input_data['front_image'] # [bs, 3, 512, 512]
        normal_F = input_data['front_nrm_img'] # [bs, 3, 512, 512]
        normal_B = input_data['back_nrm_img'] # [bs, 3, 512, 512]
        fuse_image_F=torch.cat([front_view_img,normal_F,normal_B], dim=1)
        

        multi_views={
            "image_B":normal_B,
        }
        F_plane_feat,B_plane_feat = self.image_filter(fuse_image_F, multi_views) # [bs, 32, 128, 128]) * 4
        nrm_feats = [F_plane_feat,B_plane_feat]
        return  nrm_feats
    
    def extract_map(self, input_data):
        color_feats_map = self.color_feature(input_data)
        nrm_feats_map = self.nrm_feature(input_data)
        return color_feats_map, nrm_feats_map
    
        
        


        
        

    
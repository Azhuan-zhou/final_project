"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import PIL.Image as Image
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import trimesh
import nvdiffrast.torch as dr
import nvdiffrast
import kaolin as kal
from dataset.CAPE_dataset import render_visiblity

class CustomDataset(Dataset):
    def __init__(self,root,cfg, smpl_F,device):
        self.root = root
        self.device = device
        self.img_size = cfg.img_size
        self.aug_jitter = cfg.aug_jitter
        self.aug_bg = False
        self.object_names = os.listdir(os.path.join(root,'imgs'))
        self.num_objects = len(self.object_names)
       
        

        self.transform_rgba= transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5,0.0], std=[0.5,0.5,0.5,1.0])
        ])
        self.transform_rgb= transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
        self.jitter = transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.25)

        self.F = smpl_F.to(device)
        #  set camera
        
        self.glctx = nvdiffrast.torch.RasterizeCudaContext(device=device)
        #  set camera
        look_at = torch.zeros( (2, 3), dtype=torch.float32, device=device)
        camera_up_direction = torch.tensor( [[0, 1, 0]], dtype=torch.float32, device=device).repeat(2, 1,)
        camera_position = torch.tensor( [[0, 0, 3],
                                        [0, 0, -3]], dtype=torch.float32, device=device)


        self.camera = kal.render.camera.Camera.from_args(eye=camera_position,
                                         at=look_at,
                                         up=camera_up_direction,
                                         width=self.img_size, height=self.img_size,
                                         near=-512, far=512,
                                        fov_distance=1.0, device=device)
        


    def load_npy(self,file):
        data = np.load(file, allow_pickle=True).item()
        return data


    def _add_background(self, image, mask, bg_color):
        # Random background
        bg = torch.ones_like(image) * bg_color.view(3,1,1)
        _mask = (mask<0.5).expand_as(image)
        image[_mask] = bg[_mask]
        return image

    
    def get_smplx(self, smplx_path):
        """load smplx parameters."""
        smpl_obj_path = os.path.join(smplx_path)
        smpl_obj =trimesh.load(smpl_obj_path)
        smpl_v = smpl_obj.vertices
        smpl_v = torch.from_numpy(smpl_v.astype(np.float32))
        return smpl_v
    
    
    def get_side_view_images(self,file,bg_color):
        """Load side view image."""
        side_view_rgb_names = ['color_2_masked.png','color_3_masked.png','color_4_masked.png']
        side_view_normal_names = ['normals_2_masked.png','normals_3_masked.png','normals_4_masked.png']
        rgbs = []
        normals = []
        nrm_img_path =  os.path.join(file, 'normal','normals_0_masked.png')
        nrm_img = Image.open(nrm_img_path)
        nrm_img = self.transform_rgb(nrm_img)
        for i in side_view_rgb_names:
            img_path = os.path.join(file, 'rgb',i)
            img = Image.open(img_path)
            img = self.transform_rgb(img)
            rgbs.append(img)
        for i in side_view_normal_names:
            img_path = os.path.join(file, 'normal',i)
            img = Image.open(img_path)
            img = self.transform_rgb(img)
            normals.append(img)
        return nrm_img,rgbs, normals
    
    def get_images(self,file,bg_color):
        """Load image."""
        img_path = os.path.join(file, 'rgb.png')
      

        img = Image.open(img_path)
        img = self.transform_rgba(img)
        img_mask = img[-1:,...]
        rgb = img[:-1,...]
        view_img = self._add_background(rgb, img_mask, bg_color)
        
        
        nrm_img, side_rgbs, side_normals = self.get_side_view_images(file, bg_color)
        return view_img,nrm_img,side_rgbs,side_normals
    
    
    
    
    def get_data(self, object_id):
        """Retrieve point sample."""
        
        object_name = self.object_names[object_id]
      
        object_dir = os.path.join(self.root, 'imgs',object_name)
        smpl_path = os.path.join(self.root, 'smpl',object_name+'_smplx.obj')
        smpl_v = self.get_smplx(smpl_path)
        if smpl_v.shape[0] == 1:
            smpl_v = smpl_v.squeeze(0)
       
        bg_color = torch.ones((3, 1))
        
        if self.aug_bg:
            bg_color = (torch.rand(3).float() - 0.5) / 0.5 
        
        front_image,nrm_img,side_images,side_normals = self.get_images(object_dir, bg_color)
        if self.aug_jitter:
            if torch.rand(1) > 0.5:
                front_image = self.jitter(front_image)
        
        vis_class = render_visiblity(self.camera, smpl_v, self.F, self.device, self.glctx, size=(self.img_size, self.img_size))
       
                
                
        return {

            'idx': object_name,
            'smpl_v': smpl_v,
            'vis_class': vis_class.unsqueeze(-1),
            
            'front_image': front_image,
            'side_images': side_images,
            'front_nrm_img': nrm_img,
            'back_nrm_img': side_normals[1],
            'side_normals': side_normals,
            
        }
                
    
    def __getitem__(self, idx: int):
        """Retrieve point sample."""


        return self.get_data(idx)


    
    
    def __len__(self):
        """Return length of dataset (number of _samples_)."""

        return self.num_objects
    
    
if __name__ == "__main__":
    import pickle
    import argparse
    
    from omegaconf import OmegaConf
    from dataset.CAPE_dataset import visualize_visible_vertices_3d
    from configs.misc import load_config, TrainConfig
    from torch.utils.data import DataLoader
    from dataset.THuman_dataset import project
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,default='configs/train.yaml')
    args, extras = parser.parse_known_args()
     

    # parse YAML config to OmegaConf
    cfg = load_config(args.config, cli_args=extras)
    schema = OmegaConf.structured(TrainConfig)
    cfg = OmegaConf.merge(schema, cfg)
    smpl_path = "data/body_models/smpl_data/smplx_watertight.pkl"
    with open(smpl_path, 'rb') as f:
       watertight = pickle.load(f)
       smpl_F = watertight['smpl_F']
    dataset = CustomDataset('/root/autodl-tmp/final_project/data/examples/mv',cfg, smpl_F, 'cuda')
    #(smpl_F.shape)
    print(len(dataset))
    dl =  DataLoader(dataset=dataset,batch_size=1,shuffle=True)
    sample_batch = next(iter(dl))
    smplx_v = sample_batch['smpl_v']

    front_img = sample_batch['front_image']
    side_imgs = sample_batch['side_images']
    smplx_xy = smplx_v[:,:,:2]
    smplx_zy = smplx_v[:,:,[2,1]]
    smplx_zy[...,0] = -smplx_zy[...,0]
    smplx_zy[...,1] = smplx_zy[...,1]
    import numpy as np
    import matplotlib.pyplot as plt
    F_p = smplx_xy[0]
    B_p = smplx_xy[0]
    
    L_p= smplx_zy[0]
    R_p = smplx_zy[0]
  
    project(F_p,front_img[0],'smplx_f.png')
    project(B_p,side_imgs[1][0],'smplx_b.png')
    project(L_p,side_imgs[0][0],'smplx_l.png')
    project(R_p,side_imgs[2][0],'smplx_r.png')
    
    vertices = sample_batch['smpl_v'][0]
    faces = dataset.F
    vis = sample_batch['vis_class'][0].squeeze(-1) > 0

    visualize_visible_vertices_3d(vertices, faces, vis, save_path='visible_3d.html')
    print(sample_batch['idx'])
    
    

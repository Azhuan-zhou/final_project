"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
import io 
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
import PIL.Image as Image


import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class THumanReconDataset(Dataset):
    def __init__(self, root, cfg,mode='train'):
        self.root = root
        self.mode = mode
        self.img_size = cfg.img_size
        self.num_samples = cfg.num_samples
        self.aug_jitter = cfg.aug_jitter
        if mode == 'train':
            self.aug_bg = not cfg.white_bg
        else:
            self.aug_bg = False
        self.object_names = []
        with open(os.path.join(os.path.dirname(root),'{}.txt'.format(mode)),'r') as f:
            for i in f.readlines():
                if i:
                    self.object_names.append(i.strip())
        self.num_objects = len(self.object_names)
        
        ob_imgs = os.listdir(os.path.join(root, self.object_names[0],'imgs'))
        ob_params_pts = os.path.join(root, self.object_names[0],'params','point.npy')
        self.num_views = len(ob_imgs)
        self.num_pts = self.load_npy(ob_params_pts)['xyz'].shape[0]
        

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
        


    def load_npy(self,file):
        data = np.load(file, allow_pickle=True).item()
        return data

    def _rotation_matrix(self, azimuth_deg, elevation_deg):
        # Convert degrees to radians
        theta = math.radians(azimuth_deg)
        phi = math.radians(elevation_deg)
    
        # Azimuth: Rotation about y-axis
        Ry = torch.tensor([
            [math.cos(theta), 0, math.sin(theta)],
            [0, 1, 0],
            [-math.sin(theta), 0, math.cos(theta)]
        ])
    
        # Elevation: Rotation about x-axis
        Rx = torch.tensor([
            [1, 0, 0],
            [0, math.cos(phi), -math.sin(phi)],
            [0, math.sin(phi), math.cos(phi)]
        ])
    
        # Combined rotation matrix
        R = torch.mm(Ry, Rx)
        return R


    def _add_background(self, image, mask, bg_color):
        # Random background
        bg = torch.ones_like(image) * bg_color.view(3,1,1)
        _mask = (mask<0.5).expand_as(image)
        image[_mask] = bg[_mask]
        return image
    
    def get_points(self,file,pts_id):
        """Load point cloud."""
        point_path = os.path.join(file, 'point.npy')
        data = self.load_npy(point_path)
        pts = torch.from_numpy(data['xyz'][pts_id])
        rgb = torch.from_numpy(data['rgb'][pts_id])
        nrm = torch.from_numpy(data['nrm'][pts_id])
        d = torch.from_numpy(data['d'][pts_id])
        return pts, rgb, nrm, d
    
    def get_smplx(self, file):
        """load smplx parameters."""
        smplx_path = os.path.join(file, 'smpl.npy')
        
        data = self.load_npy(smplx_path)
        smplx = torch.from_numpy(data['vertices'])
        joint_3d = torch.from_numpy(data['joints'])
        vis = torch.from_numpy(data['vis'])
        return smplx, joint_3d, vis
    
    def get_camera(self, file,view_id):
        """load camera parameters."""
        cam_path = os.path.join(file, 'camera.npy')
        data = self.load_npy(cam_path)
        cam_eva = data['cam_eva'][view_id]
        cam_azh = data['cam_azh'][view_id]
        cam_rad = data['cam_rad'][view_id]
        cam_pos = data['cam_pos'][view_id]
        return cam_eva, cam_azh, cam_rad, cam_pos
    
    
    
    def get_side_view_images(self,file):
        """Load side view image."""
        side_view_rgb_names = ['color_2_masked.png','color_3_masked.png','color_4_masked.png']
        side_view_normal_names = ['normals_2_masked.png','normals_3_masked.png','normals_4_masked.png']
        rgbs = []
        normals = []

        for i in side_view_rgb_names:
            img_path = os.path.join(file, 'rgb',i)
            img = Image.open(img_path)
            #img = affine(img)
            img = self.transform_rgb(img)
            
            rgbs.append(img)
        for i in side_view_normal_names:
            img_path = os.path.join(file, 'normal',i)
            img = Image.open(img_path)
            #img = affine(img)
            img = self.transform_rgb(img)
            normals.append(img)
        return rgbs, normals
    
    def get_images(self,file,view_id,back_id,bg_color,R):
        """Load image."""
        img_path = os.path.join(file,'imgs', 'view'+f"{view_id:02d}", 'rgb.png')
        back_img_path = os.path.join(file,'imgs', 'view'+f"{view_id:02d}", 'rgb.png')
        nrm_path = os.path.join(file,'imgs', 'view'+f"{view_id:02d}", 'nrm.png')
        back_nrm_path = os.path.join(file,'imgs', 'view'+f"{view_id:02d}", 'nrm.png')
        
        
        img = Image.open(img_path)
        img = self.transform_rgba(img)
        img_mask = img[-1:,...]
        rgb = img[:-1,...]
        view_img = self._add_background(rgb, img_mask, bg_color)
        input_size = view_img.shape[1:]
        
        back_img = Image.open(back_img_path)
        back_img = self.transform_rgba(back_img)
        back_img_mask = back_img[-1:,...]
        
        nrm = Image.open(nrm_path)
        nrm = self.transform_rgba(nrm)[:3,...]
        nrm_img = (R @ nrm.view(3, -1)).view(3, input_size[0], input_size[1]) * img_mask
        
        back_nrm = Image.open(back_nrm_path)
        back_nrm = self.transform_rgba(back_nrm)[:3,...]
        back_nrm_img = (R @ back_nrm.view(3, -1)).view(3, input_size[0], input_size[1]) * back_img_mask
        
        
        side_rgbs, side_normals = self.get_side_view_images(os.path.join(file,'imgs', 'view'+f"{view_id:02d}"))
        return view_img,img_mask,side_rgbs, nrm_img,back_nrm_img,side_normals
    
    
    
    
    def get_data(self, object_id, pts_id, view_id):
        """Retrieve point sample."""
        
        object_name = self.object_names[object_id]
        back_view_id = (view_id + self.num_views // 2 ) % self.num_views
        object_dir = os.path.join(self.root, object_name)
        params_dir = os.path.join(self.root, object_name, 'params')
        pts, rgb, nrm, d = self.get_points(params_dir,pts_id)
        smplx_v, joint_3d, vis = self.get_smplx(params_dir)
        if smplx_v.shape[0] == 1:
            smplx_v = smplx_v.squeeze(0)
        eva, azh, rad, pos = self.get_camera(params_dir,view_id)
        front_vis =vis[view_id].bool()
        back_vis = vis[back_view_id].bool()
        vis_class = torch.zeros_like(front_vis, dtype=torch.float32)
        # assign vis class to 1 if front_vis, -1 if back_vis, 0 if none
        vis_class[front_vis] = 1.0
        vis_class[back_vis] = -1.0
        
        R = self._rotation_matrix(-azh, -eva)
        
        # Transform points
        pts = pts @ R.t()
        nrm = nrm @ R.t()
        smplx_v = smplx_v @ R.t() 
        bg_color = torch.ones((3, 1))
        
        if self.aug_bg:
            bg_color = (torch.rand(3).float() - 0.5) / 0.5 
        
        front_image, _,side_images, nrm_img,back_nrm_img, side_normals = self.get_images(object_dir, view_id,back_view_id, bg_color,R)
        if self.aug_jitter:
            if torch.rand(1) > 0.5:
                front_image = self.jitter(front_image)
                
       
                
                
        return {
            'pts': pts,
            'd': d,
            'nrm': nrm,
            'rgb': rgb,
            'idx': object_name,
            'smpl_v': smplx_v,
            'vis_class': vis_class.unsqueeze(-1),
            
            'front_image': front_image,
            'side_images': side_images,
            'front_nrm_img': nrm_img,
            'back_nrm_img': back_nrm_img,
            'side_normals': side_normals,
            
        }
                
        

    def __getitem__(self, idx: int):
        """Retrieve point sample."""
      
        # points id need to be in accending order
        if self.mode == 'train':
            pts_id = np.random.randint(self.num_pts - self.num_samples, size=1)
            pts = np.arange(pts_id, pts_id + self.num_samples)
            obj_name = self.object_names[idx]
            face_path = os.path.join(self.root, obj_name, 'face_info.npy')
            views = list(self.load_npy(face_path).keys())
            view = np.random.choice(views)
            view_id = int(view[4:])
        else:
            pts =  np.arange(self.num_pts)[:self.num_samples]
            view_id = 0

        return self.get_data(idx, pts, view_id)


    
    
    def __len__(self):
        """Return length of dataset (number of _samples_)."""

        return self.num_objects

def project(xy,img,name):
    import matplotlib.pyplot as plt
    xy =torch.clip(xy, -1, 1).clone().cpu().numpy()
   
    _, img_H, img_W = img.shape
    xy[:,1 ]= -xy[:,1]  # y轴翻转
    xy_pixel = (xy + 1) / 2.0  # 归一化到[0,1]
    xy_pixel[:, 0] = xy_pixel[:, 0] * img_W  # x轴放缩
    xy_pixel[:, 1] = xy_pixel[:, 1] * img_H  # y轴放缩
    img_np = img.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    img_np = (img_np + 1) / 2.0  # 还原到 [0,1]
    img_np = np.clip(img_np, 0, 1)
    # 4. 画图
    plt.figure(figsize=(8,8))
    plt.imshow(img_np)
    plt.scatter(xy_pixel[:, 0], xy_pixel[:, 1], s=1, c='r')  # 点很小，红色
    plt.axis('off')
    plt.savefig(name, dpi=300, bbox_inches='tight')
    
if __name__ == "__main__":
    import argparse
    
    from omegaconf import OmegaConf
    
    from configs.misc import load_config, TrainConfig
    from torch.utils.data import DataLoader
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,default='configs/train.yaml')
    args, extras = parser.parse_known_args()
     

    # parse YAML config to OmegaConf
    cfg = load_config(args.config, cli_args=extras)
    schema = OmegaConf.structured(TrainConfig)
    cfg = OmegaConf.merge(schema, cfg)
    dataset = THumanReconDataset(cfg.data_root,cfg,mode='train')
    dl =  DataLoader(dataset=dataset,batch_size=1,shuffle=True)
    sample_batch = next(iter(dl))
    smplx_v = sample_batch['smpl_v']
    print(smplx_v.shape)
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
    pts = sample_batch['pts']
    pts_xy = pts[:,:,:2]
    pts_zy = pts[:,:,[2,1]]
    pts_zy[...,0] = -pts_zy[...,0]
    pts_zy[...,1] = pts_zy[...,1]
    project(F_p,front_img[0],'smplx_f.png')
    project(B_p,side_imgs[1][0],'smplx_b.png')
    project(L_p,side_imgs[0][0],'smplx_l.png')
    project(R_p,side_imgs[2][0],'smplx_r.png')
    project(pts_xy[0],front_img[0],'pts_f.png')
    project(pts_xy[0],side_imgs[1][0],'pts_b.png')     
    project(pts_zy[0],side_imgs[0][0],'pts_l.png')
    project(pts_zy[0],side_imgs[2][0],'pts_r.png')
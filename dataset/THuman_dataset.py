"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
import io 
import math
import PIL.Image as Image
import numpy as np
import os
from smplx import SMPLX
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import nvdiffrast
import kaolin as kal
import pickle
import cv2
WATERTIGHT_TEMPLATE = 'data/smplx_watertight.pkl'
class THumanReconDataset(Dataset):
    def __init__(self, root, cfg,mode='train'):
        self.root = root
        self.mode = mode
        self.smplx_root = cfg.smpl_root
        self.img_size = cfg.img_size
        self.num_samples = cfg.num_samples
        self.aug_jitter = cfg.aug_jitter
        self.aug_bg = not cfg.white_bg
        self.object_names = []
        with open(os.path.join(root,'{}.txt'.format(mode)),'r') as f:
            for i in f.readlines():
                self.object_names.append(i.strip())
        self.num_objects = len(self.object_names)
        
        ob_imgs = os.listdir(os.path.join(root, self.object_names[0],'imgs'))
        ob_params_pts = os.path.join(root, self.object_names[0],'params','points.npy')
        self.num_views = len(ob_imgs)
        self.num_pts = self.load_npy(ob_params_pts).shape[0]
        

        self.transform_rgba= transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5,0.0], std=[0.5,0.5,0.5,1.0])
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
        point_path = os.path.join(file, 'points.npy')
        data = self.load_npy(point_path)
        pts = data['xyz'][pts_id]
        rgb = data['rgb'][pts_id]
        nrm = data['nrm'][pts_id]
        d = data['d'][pts_id]
        return pts, rgb, nrm, d
    
    def get_smplx(self, file):
        """load smplx parameters."""
        smplx_path = os.path.join(file, 'smpl.npy')
        data = self.load_npy(smplx_path)
        smplx = data['vertices']
        joint_3d = data['joint_3d']
        vis = data['vis']
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
    
    def get_side_view_images(self,file,bg_color):
        """Load side view image."""
        side_view_rgb_names = ['color_0_masked.png', 'color_1_masked.png', 'color_2_masked.png','color_3_masked.png','color_4_masked.png','color_5_masked.png']
        side_view_normal_names = ['normals_0_masked.png', 'normals_1_masked.png', 'normals_2_masked.png','normals_3_masked.png','normals_4_masked.png','normals_5_masked.png']
        rgbs = []
        normals = []
        for i in side_view_rgb_names:
            img_path = os.path.join(file, 'rgb',i)
            img = Image.open(img_path)
            img = self.transform_rgba(img)
            img_mask = img[-1:,...]
            rgb = img[:-1,...]
            view_img = self._add_background(rgb, img_mask, bg_color)
            rgbs.append(view_img)
        for i in side_view_normal_names:
            img_path = os.path.join(file, 'normal',i)
            img = Image.open(img_path)
            img = self.transform_rgba(img)
            img_mask = img[-1:,...]
            rgb = img[:-1,...]
            view_img = self._add_background(rgb, img_mask, bg_color)
            normals.append(view_img)
        return rgbs, normals
    
    def get_images(self,file,view_id,back_id,bg_color,R):
        """Load image."""
        img_path = os.path.join(file, 'view'+str(view_id), 'rgb.png')
        back_img_path = os.path.join(file, 'view'+str(back_id), 'rgb.png')
        nrm_path = os.path.join(file, 'view'+str(view_id), 'nrm.png')
        back_nrm_path = os.path.join(file, 'view'+str(back_id), 'nrm.png')
        
        
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
        
        
        side_rgbs, side_normals = self.get_side_view_images(os.path.join(file, 'view'+str(view_id)), bg_color)
        return view_img,img_mask,side_rgbs, nrm_img,back_nrm_img,side_normals
    
    
    
    
    def get_data(self, object_id, pts_id, view_id):
        """Retrieve point sample."""
        
        object_name = "%04d" % object_id
        back_view_id = (view_id + self.num_views // 2 ) % self.num_views
        object_dir = os.path.join(self.root, object_name)
        params_dir = os.path.join(self.root, object_name, 'params')
        pts, rgb, nrm, d = self.get_points(params_dir,pts_id)
        smplx_v, joint_3d, vis = self.get_smplx(params_dir)
        eva, azh, rad, pos = self.get_camera(params_dir,view_id)
        front_vis = vis[view_id].astype(bool)
        back_vis = vis[back_view_id].astype(bool)
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
        
        front_image, front_image_mask,side_images, nrm_img,back_nrm_img, side_normals = self.get_images(object_dir, view_id,back_view_id, bg_color,R)
        if self.aug_jitter:
            if torch.rand(1) > 0.5:
                front_image = self.jitter(front_image)
                
       
                
                
        return {
            'pts': pts,
            'd': d,
            'nrm': nrm,
            'rgb': rgb,
            'idx': object_id,
            'smpl_v': smplx_v,
            'vis_class': vis_class.unsqueeze(-1),
            
            'front_image': front_image,
            'front_mask': front_image_mask,
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
            view_id = int(np.random.randint(self.num_views, size=1))
        else:
            pts =  np.arange(self.num_pts)
            view_id = 0

        return self.get_data(idx, pts, view_id)


    
    
    def __len__(self):
        """Return length of dataset (number of _samples_)."""

        return self.num_objects
    
    
def get_visibility(xy, z, faces):
    """get the visibility of vertices

    Args:
        xy (torch.tensor): [N,2]
        z (torch.tensor): [N,1]
        faces (torch.tensor): [N,3]
        size (int): resolution of rendered image
    """

    xyz = torch.cat((xy, -z), dim=1)
    xyz = (xyz + 1.0) / 2.0
    faces = faces.long()

    rasterizer = Pytorch3dRasterizer(image_size=2**12)
    meshes_screen = Meshes(verts=xyz[None, ...], faces=faces[None, ...])
    raster_settings = rasterizer.raster_settings

    pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
        meshes_screen,
        image_size=raster_settings.image_size,
        blur_radius=raster_settings.blur_radius,
        faces_per_pixel=raster_settings.faces_per_pixel,
        bin_size=raster_settings.bin_size,
        max_faces_per_bin=raster_settings.max_faces_per_bin,
        perspective_correct=raster_settings.perspective_correct,
        cull_backfaces=raster_settings.cull_backfaces,
    )

    vis_vertices_id = torch.unique(faces[torch.unique(pix_to_face), :])
    vis_mask = torch.zeros(size=(z.shape[0], 1))
    vis_mask[vis_vertices_id] = 1.0

    # print("------------------------\n")
    # print(f"keep points : {vis_mask.sum()/len(vis_mask)}")

    return vis_mask


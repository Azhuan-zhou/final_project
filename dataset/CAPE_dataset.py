import io 
import math
import PIL.Image as Image
import numpy as np
import os
from models.smpl.smpl_numpy import SMPL
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import nvdiffrast
import kaolin as kal
import cv2

class TestReconDataset(Dataset):
    def __init__(self,root,img_size,erode_iter,watertight,device):
        self.root = root
        self.device = device
        self.image_folder = os.path.join(self.root, 'images')
        self.smplx_folder = os.path.join(self.root, 'smplx')
        self.smpl_folder = os.path.join(self.root, 'smpl')

        self.subject_names = [x for x in sorted(os.listdir(self.image_folder))]
        
        self.subject_list = [os.path.join(self.image_folder,x,'rgb.png') for x in self.subject_names]
        self.subject_back_list = [os.path.join(self.image_folder,x,'rgb_back.png') for x in self.subject_names]
        self.side_view_folder = [os.path.join(self.image_folder,x,'rgb') for x in self.subject_names]
        self.smplx_list = [os.path.join(self.smplx_folder, x+'.obj') for x in self.subject_names]
        
        assert len(self.subject_list) == len(self.smplx_list) == len( self.side_view_folder)
        
        self.img_size = img_size
        self.erode_iter = erode_iter

        
        self.num_subjects = len(self.subject_list)
        self.img_size = img_size
        self.erode_iter = erode_iter

        self.F = watertight['smpl_F'].to(device)
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
        
        self.glctx = nvdiffrast.torch.RasterizeCudaContext(device=device)
        self.transform= transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        self.to_tensor= transforms.Compose([
            transforms.ToTensor(),
        ]) 
      
        
        
        smpl_model = SMPL(sex='neutral', model_dir='data/body_models/smpl/SMPL_NEUTRAL.pkl')
        self.big_pose_params = self.get_big_pose_params()
        t_vertices, _ = smpl_model(self.big_pose_params['poses'], self.big_pose_params['shapes'].reshape(-1))
        self.t_vertices = t_vertices.astype(np.float32)
        min_xyz = np.min(self.t_vertices, axis=0)
        max_xyz = np.max(self.t_vertices, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        min_xyz[2] -= 0.1
        max_xyz[2] += 0.1
        self.t_world_bounds = np.stack([min_xyz, max_xyz], axis=0)
        
        
    def render_visiblity(self, camera, V, F, size=(1024, 1024)):
        
        vertices_camera = camera.extrinsics.transform(V)
        face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(
                            vertices_camera, F)
        face_normals_z = kal.ops.mesh.face_normals(face_vertices_camera,unit=True)[..., -1:].contiguous()
        proj = camera.projection_matrix()[0:1]
        homogeneous_vecs = kal.render.camera.up_to_homogeneous(
            vertices_camera
        )[..., None]
        vertices_clip = (proj @ homogeneous_vecs).squeeze(-1)
        rast = nvdiffrast.torch.rasterize(
            self.glctx, vertices_clip, F.int(),
            size, grad_db=False
        )
        rast0 = torch.flip(rast[0], dims=(1,))
        face_idx = (rast0[..., -1:].long() - 1).contiguous()
        # assign visibility to 1 if face_idx >= 0
        vv = []
        for i in range(rast0.shape[0]):
            vis = torch.zeros((F.shape[0],), dtype=torch.bool, device=self.device)
            for f in range(F.shape[0]):
                vis[f] = 1 if torch.any(face_idx[i] == f) else 0
            vv.append(vis)

        front_vis = vv[0].bool()
        back_vis = vv[1].bool()
        vis_class = torch.zeros((F.shape[0], 1), dtype=torch.float32)
        vis_class[front_vis] = 1.0
        vis_class[back_vis] = -1.0

        return vis_class

    def add_background(self, image, mask, color=[0.0, 0.0, 0.0]):
        # Random background
        bg_color = torch.tensor(color).float()
        bg = torch.ones_like(image) * bg_color.view(3,1,1)
        _mask = (mask<0.5).expand_as(image)
        image[_mask] = bg[_mask]
        return image
    
    
    def erode_mask(self, mask, kernal=(5,5), iter=1):
        mask = torch.from_numpy(cv2.erode(mask[0].numpy(), np.ones(kernal, np.uint8), iterations=iter)).float().unsqueeze(0)
        return mask

    def load_npy(self,file):
        data = np.load(file, allow_pickle=True).item()
        return data
    
    def get_side_view_images(self,file,bg_color):
        """Load side view image."""
        side_view_rgb_names = ['color_0_masked.png', 'color_1_masked.png', 'color_2_masked.png','color_3_masked.png']
        side_view_normal_names = ['normals_0_masked.png', 'normals_1_masked.png', 'normals_2_masked.png','normals_3_masked.png']
        rgbs = []
        normals = []
        for i in side_view_rgb_names:
            img_path = os.path.join(file, 'rgb',i)
            img = Image.open(img_path)
            img = self.transform(img)
            img_mask = img[-1:,...]
            rgb = img[:-1,...]
            view_img = self.add_background(rgb, img_mask, bg_color)
            rgbs.append(view_img)
        for i in side_view_normal_names:
            img_path = os.path.join(file, 'normal',i)
            img = Image.open(img_path)
            img = self.transform(img)
            img_mask = img[-1:,...]
            rgb = img[:-1,...]
            view_img = self.add_background(rgb, img_mask, bg_color)
            normals.append(view_img)
        return rgbs, normals
    
    def get_images(self,subject_id,bg_color,R):
        """Load image."""
        img_path = self.subject_list[subject_id]
        back_img_path = self.subject_back_list[subject_id]
        nrm_path = os.path.join(file, 'view'+str(view_id), 'nrm.png')
        back_nrm_path = os.path.join(file, 'view'+str(back_id), 'nrm.png')
        
        
        img = Image.open(img_path)
        img = self.transform(img)
        img_mask = img[-1:,...]
        rgb = img[:-1,...]
        view_img = self.add_background(rgb, img_mask, bg_color)
        input_size = view_img.shape[1:]
        
        back_img = Image.open(back_img_path)
        back_img = self.transform(back_img)
        back_img_mask = back_img[-1:,...]
        
        nrm = Image.open(nrm_path)
        nrm = self.transform_rgba(nrm)[:3,...]
        nrm_img = (R @ nrm.view(3, -1)).view(3, input_size[0], input_size[1]) * img_mask
        
        back_nrm = Image.open(back_nrm_path)
        back_nrm = self.transform_rgba(back_nrm)[:3,...]
        back_nrm_img = (R @ back_nrm.view(3, -1)).view(3, input_size[0], input_size[1]) * back_img_mask
        
        
        side_rgbs, side_normals = self.get_side_view_images(os.path.join(file, 'view'+str(view_id)), bg_color)
        return view_img,img_mask,side_rgbs, nrm_img,back_nrm_img,side_normals
    
    
    def get_big_pose_params(self):

        big_pose_params = {}
        # big_pose_params = copy.deepcopy(params)
        big_pose_params['R'] = np.eye(3).astype(np.float32)
        big_pose_params['Th'] = np.zeros((1,3)).astype(np.float32)
        big_pose_params['shapes'] = np.zeros((1,10)).astype(np.float32)
        big_pose_params['poses'] = np.zeros((1,72)).astype(np.float32)
        big_pose_params['poses'][0, 5] = 45/180*np.array(np.pi)
        big_pose_params['poses'][0, 8] = -45/180*np.array(np.pi)
        big_pose_params['poses'][0, 23] = -30/180*np.array(np.pi)
        big_pose_params['poses'][0, 26] = 30/180*np.array(np.pi)
        return big_pose_params
    
    def get_data(self, subject_id, pts_id, view_id):
        """Retrieve point sample."""
        object_name = "%04d" % subject_id
        back_view_id = (view_id + self.num_views // 2 ) % self.num_views
        object_dir = os.path.join(self.root, object_name)
        params_dir = os.path.join(self.root, object_name, 'params')
        pts, rgb, nrm, d = self.get_points(params_dir,pts_id)
        smpl_v, joint_3d, vis = self.get_smpl(params_dir)
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
        smpl_v = smpl_v @ R.t() 
        bg_color = torch.ones((3, 1))
        
        if self.aug_bg:
            bg_color = (torch.rand(3).float() - 0.5) / 0.5 
        
        front_image, front_image_mask,side_images, nrm_img,back_nrm_img, side_normals = self.get_images(object_dir, view_id,back_view_id, bg_color,R)
        if self.aug_jitter:
            if torch.rand(1) > 0.5:
                front_image = self.jitter(front_image)
                
                
        smpl_path = os.path.join(self.smpl_root,'{}_smpl.pkl'.format(object_name))
        smpl_data = np.load(smpl_path, allow_pickle=True)
        obs_param = {
            'pose': smpl_data['betas'],
            'shape': smpl_data['body_pose']
        }
                
                
        return {
            'pts': pts,
            'd': d,
            'nrm': nrm,
            'rgb': rgb,
            'idx': subject_id,
            'smpl_v': smpl_v,
            'vis_class': vis_class.unsqueeze(-1),
            
            'front_image': front_image,
            'front_mask': front_image_mask,
            'side_images': side_images,
            'front_nrm_img': nrm_img,
            'back_nrm_img': back_nrm_img,
            'side_normals': side_normals,
            
            # canonical space
            't_params': self.big_pose_params,
            't_vertices': self.t_vertices,
            't_world_bounds': self.t_world_bounds,
            
            
            #obs
            'obs_params': obs_param
            
        }
                
        

    def __getitem__(self, idx: int):
        """Retrieve point sample."""
      
        # points id need to be in accending order
        pts_id = np.random.randint(self.num_pts - self.num_samples, size=1)
        pts = np.arange(pts_id, pts_id + self.num_samples)

        view_id = int(np.random.randint(self.num_views, size=1))

        return self.get_data(idx, pts, view_id)


    
    
    def __len__(self):
        """Return length of dataset (number of _samples_)."""

        return self.num_subjects

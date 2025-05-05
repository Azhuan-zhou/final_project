import io 
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import PIL.Image as Image
import PIL
from typing import  Tuple, Optional
import numpy as np
import os
from smplx import SMPLX
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import nvdiffrast
import kaolin as kal
import cv2
from dataset.mv_dataset import add_margin
from script.detect_face import detect_face 
import trimesh
import nvdiffrast.torch as dr

def projection(points, calib):
    if torch.is_tensor(points):
        calib = torch.as_tensor(calib) if not torch.is_tensor(calib) else calib
        return torch.mm(calib[:3, :3], points.T).T + calib[:3, 3]
    else:
        return np.matmul(calib[:3, :3], points.T).T + calib[:3, 3]
    
class SingleImageDataset(Dataset):
    def __init__(self,
        root_dir: str,
        num_views: int,
        img_wh: Tuple[int, int],
        bg_color: str,
        objects_txt:str,
        margin_size: int = 0,
        single_image: Optional[PIL.Image.Image] = None,
        prompt_embeds_path: Optional[str] = None,
        crop_size: Optional[int] = 720,
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = root_dir
        self.num_views = num_views
        self.img_wh = img_wh
        self.margin_size = margin_size
        self.bg_color = bg_color
        self.crop_size = crop_size
        self.crop_size = crop_size
        # load all images
       
        if single_image is None:
            file_list = []
            mask_list = []
            object_names = []
            # Get a list of all files in the directory
            with open(objects_txt,'r') as f:
                for i in f.readlines():
                    if i:
                        object_names.append(i.strip())
            for object_name in object_names:
                img_path = os.path.join(self.root_dir, 'view',object_name,'front.png')
                mask_path = os.path.join(self.root_dir, 'view',object_name,'mask.png')
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    file_list.append(img_path)
                    mask_list.append(mask_path)
            # Filter the files that end with .png or .jpg
            self.file_list = file_list
        else:
            self.file_list = [single_image]

 
            
        try:
            normal_prompt_embedding = torch.load(f'{prompt_embeds_path}/normal_embeds.pt')
            color_prompt_embedding = torch.load(f'{prompt_embeds_path}/clr_embeds.pt')
            if self.num_views == 7:
                self.normal_text_embeds = normal_prompt_embedding
                self.color_text_embeds = color_prompt_embedding
            elif self.num_views == 5:
                self.normal_text_embeds = torch.stack([normal_prompt_embedding[0], normal_prompt_embedding[2], normal_prompt_embedding[3], normal_prompt_embedding[4], normal_prompt_embedding[6]] , 0)
                self.color_text_embeds = torch.stack([color_prompt_embedding[0], color_prompt_embedding[2], color_prompt_embedding[3], color_prompt_embedding[4], color_prompt_embedding[6]] , 0)
        except:
            self.color_text_embeds = torch.load(f'{prompt_embeds_path}/embeds.pt')
            self.normal_text_embeds = None

    def __len__(self):
        return len(self.file_list)

        

    def get_bg_color(self):
        if self.bg_color == 'white':
            bg_color = np.array([1., 1., 1.], dtype=np.float32)
        elif self.bg_color == 'black':
            bg_color = np.array([0., 0., 0.], dtype=np.float32)
        elif self.bg_color == 'gray':
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif self.bg_color == 'random':
            bg_color = np.random.rand(3)
        elif isinstance(self.bg_color, float):
            bg_color = np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color
    
    
        
    def load_image(self, img_path, bg_color, return_type='np', Imagefile=None):
        # pil always returns uint8
        if Imagefile is None:
            image_input = Image.open(img_path)
        else:
            image_input = Imagefile
        image_size = self.img_wh[0]
        
        if self.crop_size!=-1:
            alpha_np = np.asarray(image_input)[:, :, 3]
            coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
            min_x, min_y = np.min(coords, 0)
            max_x, max_y = np.max(coords, 0)
            ref_img_ = image_input.crop((min_x, min_y, max_x, max_y))
            h, w = ref_img_.height, ref_img_.width
            scale = self.crop_size / max(h, w)
            h_, w_ = int(scale * h), int(scale * w)
            ref_img_ = ref_img_.resize((w_, h_))
            image_input = add_margin(ref_img_, size=image_size)
        else:
            image_input = add_margin(image_input, size=max(image_input.height, image_input.width))
            image_input = image_input.resize((image_size, image_size))
        
        # img = scale_and_place_object(img, self.scale_ratio)
        img = np.array(image_input)
        img = img.astype(np.float32) / 255. # [0, 1]
        assert img.shape[-1] == 4 # RGBA

        alpha = img[...,3:4]
        
        img = img[...,:3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
            alpha = torch.from_numpy(alpha)
        else:
            raise NotImplementedError
        
        return img, alpha
    
    def load_face(self, img_path, bg_color, return_type='np', Imagefile=None):
        # pil always returns uint8
        if Imagefile is None:
            image_input = Image.open(img_path)
        else:
            image_input = Imagefile
        image_size = self.img_wh[0]

        if self.crop_size!=-1:
            alpha_np = np.asarray(image_input)[:, :, 3]
            coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
            min_x, min_y = np.min(coords, 0)
            max_x, max_y = np.max(coords, 0)
            ref_img_ = image_input.crop((min_x, min_y, max_x, max_y))
            h, w = ref_img_.height, ref_img_.width
            scale = self.crop_size / max(h, w)
            h_, w_ = int(scale * h), int(scale * w)
            ref_img_ = ref_img_.resize((w_, h_))
            image_input = add_margin(ref_img_, size=image_size)
        else:
            image_input = add_margin(image_input, size=max(image_input.height, image_input.width))
            image_input = image_input.resize((image_size, image_size))

        image_input = image_input.crop((256, 0, 512, 256)).resize((self.img_wh[0], self.img_wh[1]))
        
        # img = scale_and_place_object(img, self.scale_ratio)
        img = np.array(image_input)
        img = img.astype(np.float32) / 255. # [0, 1]
        assert img.shape[-1] == 4 # RGBA

        alpha = img[...,3:4]
        img = img[...,:3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
            alpha = torch.from_numpy(alpha)
        else:
            raise NotImplementedError
        
        return img, alpha

    
    def process_face(self,img_path, bbox, bg_color, normal_path=None, w2c=None):
        image = Image.open(img_path)
        bbox_w, bbox_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if bbox_w > bbox_h:
            bbox[1] -= (bbox_w - bbox_h) // 2
            bbox[3] += (bbox_w - bbox_h) // 2
        else:
            bbox[0] -= (bbox_h - bbox_w) // 2
            bbox[2] += (bbox_h - bbox_w) // 2
        bbox[0:2] -= 20
        bbox[2:4] += 20
        bbox[1] -= 20
        bbox[3] += 20
        image = image.crop(bbox)
         
        h, w = image.height, image.width

        scale = self.crop_size / max(h, w)
        h_, w_ = int(scale * h), int(scale * w)
        image = image.resize((w_, h_))
        image = np.array(image) / 255.
        img, alpha = image[:, :, :3], image[:, :, 3:4]
        img = img * alpha + bg_color * (1 - alpha)
        
        padded_img = np.full((self.img_wh[0], self.img_wh[1], 3), bg_color, dtype=np.float32)
        dx = (self.img_wh[0] - self.crop_size) // 2
        dy = (self.img_wh[1] - self.crop_size) // 2
        padded_img[dy:dy+h_, dx:dx+w_] = img
        padded_img = torch.from_numpy(padded_img).permute(2,0,1)
        
        return padded_img
    
    def __getitem__(self, index):
        bg_color = self.get_bg_color()
        file_path = self.file_list[index]
        img = Image.open(file_path).convert('RGB')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        boxes, probs = detect_face(img.to(device))
        max_idx = np.argmax(probs)
        if boxes is not None:
            bbox = boxes[max_idx] 
        else: 
            bbox = None
        if bbox is not None:
            face = self.process_face(file_path, bbox.astype(np.int32), bg_color)
        else:
            face,_ =  self.load_face(file_path, bg_color, return_type='pt')
            face = face.permute(2, 0, 1)
        image, alpha = self.load_image(file_path, bg_color, return_type='pt')
        
        view_path = os.path.dirname(file_path)
       
        img_tensors_in = [
            image.permute(2, 0, 1)
        ] * (self.num_views-1) + [
            face
        ]
        img_tensors_in = torch.stack(img_tensors_in, dim=0).float() # (Nv, 3, H, W)
        
        normal_prompt_embeddings = self.normal_text_embeds if hasattr(self, 'normal_text_embeds') else None
        color_prompt_embeddings = self.color_text_embeds if hasattr(self, 'color_text_embeds') else None
        if normal_prompt_embeddings is None:
            out =  {
            'imgs_in': img_tensors_in,
            'color_prompt_embeddings': color_prompt_embeddings,
            'filename': view_path,
            }
        else:
            out =  {
            'imgs_in': img_tensors_in,
            'normal_prompt_embeddings': normal_prompt_embeddings,
            'color_prompt_embeddings': color_prompt_embeddings,
            'filename': view_path,
            }
        return out
    
    
class CapeReconDataset(Dataset):
    def __init__(self,root,cfg, smpl_F,device):
        self.root = root
        self.img_size = cfg.img_size
        self.device = device
        
        self.image_folder = os.path.join(self.root, 'view')
        self.smpl_folder = os.path.join(self.root, 'smpl')

        self.object_names = []
        objects_txt = os.path.join(root,'test150.txt')
        with open(objects_txt,'r') as f:
            for i in f.readlines():
                if i:
                    self.object_names.append(i.strip())
        [x for x in sorted(os.listdir(self.image_folder))]
        
        self.image_list = [os.path.join(self.image_folder,x) for x in self.object_names]
       
        self.smpl_list = [os.path.join(self.smpl_folder, x) for x in self.object_names]
        
        
        assert len(self.image_list  ) == len(self.smpl_list)
        
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

        
        self.num_objects = len(self.image_list)

        self.F = smpl_F.to(device)
        #  set camera
        
        self.glctx = nvdiffrast.torch.RasterizeCudaContext(device=device)
        
        self.transform_rgba= transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5,0.0], std=[0.5,0.5,0.5,1.0])
        ])
        
        self.transform_rgb = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

      
        
        
        
        
    def load_calib(self, calib_path):
        calib_data = np.loadtxt(calib_path, dtype=float)
        extrinsic = calib_data[:4, :4]
        intrinsic = calib_data[4:8, :4]
        calib_mat = np.matmul(intrinsic, extrinsic)
        intrinsic= torch.from_numpy(intrinsic).float()
        extrinsic = torch.from_numpy(extrinsic).float()
        calib_mat = torch.from_numpy(calib_mat).float()
        return {'calib': calib_mat,"extrinsic": extrinsic, "intrinsic": intrinsic}
        

    def add_background(self, image, mask, color=[0.0, 0.0, 0.0]):
        # Random background
        bg_color = torch.tensor(color).float()
        bg = torch.ones_like(image) * bg_color.view(3,1,1)
        _mask = (mask<0.5).expand_as(image)
        image[_mask] = bg[_mask]
        return image

    def get_smpl(self,file):
        smpl_obj_path = os.path.join(file, 'smpl.obj')
        smplx_param = os.path.join(file, 'smpl_param.npz')
        J_0 = np.load(os.path.join(file, 'smpl_joints.npy'))[0]
        param = np.load(smplx_param,allow_pickle=True)
        transl = param['transl']
        smpl_obj =trimesh.load(smpl_obj_path, device=self.device)
        smpl_v = smpl_obj.vertices.astype(np.float32) - np.array(transl, dtype=np.float32) + J_0
        smpl_v = torch.from_numpy(smpl_v)
        return smpl_v
    
    def get_side_view_images(self,file,bg_color):
        """Load side view image."""
        side_view_rgb_names = [ 'color_2_masked.png', 'color_3_masked.png','color_4_masked.png']
        side_view_normal_names = [ 'normals_2_masked.png', 'normals_3_masked.png','normals_4_masked.png']
        rgbs = []
        normals = []
        for i in side_view_rgb_names:
            img_path = os.path.join(file,'rgb',i)
            img = Image.open(img_path)
            img = self.transform_rgb(img)
            rgbs.append(img)
        for i in side_view_normal_names:
            img_path = os.path.join(file, 'normal',i)
            img = Image.open(img_path)
            img = self.transform_rgb(img)

            normals.append(img)
        return rgbs, normals
    
    def get_images(self,object_id,bg_color):
        """Load image."""
        img_dir = self.image_list[object_id]
        img_path = os.path.join(img_dir,'front.png')
        nrm_path = os.path.join(img_dir, 'normal_F.png')
        back_nrm_path = os.path.join(img_dir, 'normal_B.png')
        
        
        img = Image.open(img_path)
        img = self.transform_rgba(img)
        img_mask = img[-1:,...]
        rgb = img[:-1,...]
        view_img = self.add_background(rgb, img_mask, bg_color)
        input_size = view_img.shape[1:]
        
        
        nrm = Image.open(nrm_path)
        nrm = self.transform_rgba(nrm)[:3,...]
        nrm_img = nrm * img_mask
        
        back_nrm = Image.open(back_nrm_path)
        back_nrm = self.transform_rgba(back_nrm)[:3,...]
        back_nrm_img = back_nrm * img_mask
        side_rgbs, side_normals = self.get_side_view_images(img_dir, bg_color)
        return view_img,img_mask,side_rgbs, nrm_img,back_nrm_img,side_normals
    
    
   
    
    def get_data(self, object_id):
        """Retrieve point sample."""
        object_name = self.object_names[object_id]
        bg_color = torch.ones((3, 1)) # white
        calib = self.load_calib(os.path.join(self.image_list[object_id], 'calib.txt'))
        
        front_image, front_image_mask,side_images, nrm_img,back_nrm_img, side_normals = self.get_images(object_id,bg_color)
        smpl_v = self.get_smpl(self.smpl_list[object_id])
        vis_class = render_visiblity(self.camera,smpl_v, self.F, self.device, self.glctx, size=(self.img_size, self.img_size))

        
     
                
                
        return {
            'smpl_v': smpl_v,
            'vis_class': vis_class.unsqueeze(-1),
            'idx': object_name,
            
            'front_image': front_image,
            'front_mask': front_image_mask,
            'side_images': side_images,
            'front_nrm_img': nrm_img,
            'back_nrm_img': back_nrm_img,
            'side_normals': side_normals,
        }
                
        

    def __getitem__(self, idx: int):
        """Retrieve point sample."""

        return self.get_data(idx)


    
    
    def __len__(self):
        """Return length of dataset (number of _samples_)."""

        return self.num_objects



def render_visiblity(camera, V, F,device,glctx, size=(1024, 1024)):
    V = V.to(device)
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
        glctx, vertices_clip, F.int(),
        size, grad_db=False
    )
    rast0 = torch.flip(rast[0], dims=(1,))
    face_idx = (rast0[..., -1:].long() - 1).contiguous()
    # assign visibility to 1 if face_idx >= 0
    vv = []
    for i in range(rast0.shape[0]):
        vis = torch.zeros((F.shape[0],), dtype=torch.bool, device=device)
        for f in range(F.shape[0]):
            vis[f] = 1 if torch.any(face_idx[i] == f) else 0
        vv.append(vis)
    front_vis = vv[0].bool()
    back_vis = vv[1].bool()
    vis_class = torch.zeros((F.shape[0], 1), dtype=torch.float32)
    vis_class[front_vis] = 1.0
    vis_class[back_vis] = -1.0

    return vis_class.squeeze(-1)

import plotly.graph_objs as go
import plotly.offline as offline

def visualize_visible_vertices_3d(vertices, faces, vis, save_path='vis_3d.html'):
    """
    vertices: [N, 3] (Tensor or ndarray)
    faces: [F, 3] (Tensor or ndarray)
    vis: [F] bool Tensor or ndarray
    save_path: output html path
    """
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()
    if isinstance(vis, torch.Tensor):
        vis = vis.detach().cpu().numpy()
    if vis.shape[0] == 1:
        vis = vis.squeeze(0)
    print(vis.shape)
    print("可见面片数量:", vis.sum().item())
    # 获取可见的面片索引
    visible_faces = faces[vis]

    # 使用三角面片绘制 mesh
    mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=visible_faces[:, 0],
        j=visible_faces[:, 1],
        k=visible_faces[:, 2],
        color='lightblue',
        opacity=0.9,
        name='Visible Mesh'
    )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )

    fig = go.Figure(data=[mesh], layout=layout)
    offline.plot(fig, filename=save_path, auto_open=False)
    
if __name__ == "__main__":
    import pickle
    import argparse
    
    from omegaconf import OmegaConf
    
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
    smpl_path = "/home/yqw/home/zsh/final_project/data/body_models/smpl/SMPL_NEUTRAL.pkl"
    with open(smpl_path, 'rb') as f:
       watertight = pickle.load(f, encoding='latin1')
       smpl_F = watertight['f']
       smpl_F = torch.as_tensor(smpl_F.astype(np.int32)).long()
    dataset = CapeReconDataset('/home/yqw/home/zsh/final_project/data/cape',cfg, smpl_F, 'cuda:3')
    #(smpl_F.shape)
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

    #visualize_visible_vertices_3d(vertices, faces, vis, save_path='visible_3d.html')
    print(sample_batch['idx'])
    
       
    
    
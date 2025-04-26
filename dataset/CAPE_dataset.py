import io 
import math
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
        self.side_view_folder = [os.path.join(self.image_folder,x,'rgb') for x in self.object_names]
       
        self.smpl_list = [os.path.join(self.smpl_folder, x) for x in self.object_names]
        
        
        assert len(self.image_list  )  == len( self.side_view_folder) == len(self.smpl_list)
        
        

        
        self.num_objects = len(self.image_list)

        self.F = smpl_F.to(device)
        #  set camera
        
        self.glctx = nvdiffrast.torch.RasterizeCudaContext(device=device)
        
        self.transform_rgba= transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5,0.0], std=[0.5,0.5,0.5,1.0])
        ])

      
        
        
        
        
    def load_calib(self, calib_path):
        calib_data = np.loadtxt(calib_path, dtype=float)
        extrinsic = calib_data[:4, :4]
        intrinsic = calib_data[4:8, :4]
        calib_mat = np.matmul(intrinsic, extrinsic)
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
        smpl_obj =trimesh.load(smpl_obj_path, device=self.device)
        smpl_v = smpl_obj.vertices
        return smpl_v=
    
    def get_side_view_images(self,file,bg_color):
        """Load side view image."""
        side_view_rgb_names = ['color_0_masked.png', 'color_2_masked.png', 'color_3_masked.png','color_4_masked.png']
        side_view_normal_names = ['normals_0_masked.png', 'normals_2_masked.png', 'normals_3_masked.png','normals_4_masked.png']
        rgbs = []
        normals = []
        for i in side_view_rgb_names:
            img_path = os.path.join(file, 'rgb',i)
            img = Image.open(img_path)
            img = self.transform_rgba(img)
            img_mask = img[-1:,...]
            rgb = img[:-1,...]
            view_img = self.add_background(rgb, img_mask, bg_color)
            rgbs.append(view_img)
        for i in side_view_normal_names:
            img_path = os.path.join(file, 'normal',i)
            img = Image.open(img_path)
            img = self.transform_rgba(img)
            img_mask = img[-1:,...]
            rgb = img[:-1,...]
            view_img = self.add_background(rgb, img_mask, bg_color)
            normals.append(view_img)
        return rgbs, normals
    
    def get_images(self,object_id,bg_color):
        """Load image."""
        img_dir = self.image_list[object_id]
        side_view_dir = self.side_view_folder[object_id]
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
        side_rgbs, side_normals = self.get_side_view_images(side_view_dir, bg_color)
        return view_img,img_mask,side_rgbs, nrm_img,back_nrm_img,side_normals
    
    
   
    
    def get_data(self, object_id):
        """Retrieve point sample."""
        object_name = self.object_names[object_id]
        bg_color = torch.ones((3, 1)) # white
        calib = self.load_calib(os.path.join(self.image_list[object_id], 'calib.txt'))
        
        front_image, front_image_mask,side_images, nrm_img,back_nrm_img, side_normals = self.get_images(object_id,bg_color)
        smpl_v,smpl_param = self.get_smpl(self.smpl_list[object_id])
        vis = render_visiblity(smpl_v, self.F, calib['extrinsic'], calib['intrinsic'], self.img_size,self.device,self.glctx)
        vis_class = torch.full_like(vis, fill_value=-1.0, dtype=torch.float32)
        front_vis = vis.bool()
        vis_class[front_vis] = 1.0
        
        smpl_v =projection(smpl_v, calib['calib'])
        
     
                
                
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

def build_projection_matrix_single(K, img_size, near=0.1, far=100.0):
    """
    K: [3, 3]
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    W = H = img_size

    proj = torch.zeros((4, 4), device=K.device)
    proj[0, 0] = 2 * fx / W
    proj[1, 1] = 2 * fy / H
    proj[0, 2] = 1 - 2 * cx / W
    proj[1, 2] = 2 * cy / H - 1
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2 * far * near / (far - near)
    proj[3, 2] = -1
    return proj


def rasterize_single(V, F, extrinsic, K, img_size, glctx):
    """
    V: [N, 3] 
    F: [F, 3] 
    extrinsic: [4, 4] or [3, 4]
    K: [3, 3] 
    img_size: int
    glctx: nvdiffrast context
    """
    N = V.shape[0]
    device = V.device

    V_h = torch.cat([V, torch.ones((N, 1), device=device)], dim=-1)  # [N, 4]

    if extrinsic.shape == (3, 4):
        extrinsic_h = torch.eye(4, device=device)
        extrinsic_h[:3, :] = extrinsic
    else:
        extrinsic_h = extrinsic
    V_cam = V_h @ extrinsic_h.T  # [N, 4]
    proj = build_projection_matrix_single(K, img_size)  # [4, 4]
    V_clip = V_cam @ proj.T  # [N, 4]

    V_ndc = V_clip[:, :3] / (V_clip[:, 3:] + 1e-8)  # [N, 3]

    rast_out = dr.rasterize(glctx, V_ndc.unsqueeze(0), F.int(), (img_size, img_size), grad_db=False)[0]  # [1, H, W, 4]

    rast_out = torch.flip(rast_out, dims=[1])  # flip H 

    hard_mask = rast_out[..., -1:] != 0  # [1, H, W, 1]

    return rast_out, hard_mask


def render_visiblity(smplx_V, F,extrinsic,K,img_size,device,glctx):
    smplx_V = smplx_V.to(device)
    F = F.to(device)
    rast0, _ = rasterize_single(smplx_V, F,extrinsic,K,img_size, glctx)

    face_idx = (rast0[..., -1:].long() - 1).contiguous()
    # assign visibility to 1 if face_idx >= 0
    vv = []
    for i in range(rast0.shape[0]):
        vis = torch.zeros((F.shape[0],), dtype=torch.bool,device=device)
        for f in range(F.shape[0]):
            vis[f] = 1 if torch.any(face_idx[i] == f) else 0
        vv.append(vis)
    vis = torch.stack(vv, dim=0)

    return vis

if __name__ == "__main__":
    dataset = CapeReconDataset(root='/home/yqw/home/zsh/final_project/data/cape',cfg=None, smpl_F=None,device='cuda')

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import  Tuple, Optional
import cv2
import random
import os
import PIL
from icecream import ic
import pdb
def add_margin(pil_img, color=0, size=256):
    width, height = pil_img.size
    result = Image.new(pil_img.mode, (size, size), color)
    result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
    return result

def scale_and_place_object(image, scale_factor):
    assert np.shape(image)[-1]==4  # RGBA

    # Extract the alpha channel (transparency) and the object (RGB channels)
    alpha_channel = image[:, :, 3]

    # Find the bounding box coordinates of the object
    coords = cv2.findNonZero(alpha_channel)
    x, y, width, height = cv2.boundingRect(coords)

    # Calculate the scale factor for resizing
    original_height, original_width = image.shape[:2]

    if width > height:
        size = width
        original_size = original_width
    else:
        size = height
        original_size = original_height

    scale_factor = min(scale_factor, size / (original_size+0.0))

    new_size = scale_factor * original_size
    scale_factor = new_size / size

    # Calculate the new size based on the scale factor
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    center_x = original_width // 2
    center_y = original_height // 2

    paste_x = center_x - (new_width // 2)
    paste_y = center_y - (new_height // 2)

    # Resize the object (RGB channels) to the new size
    rescaled_object = cv2.resize(image[y:y+height, x:x+width], (new_width, new_height))

    # Create a new RGBA image with the resized image
    new_image = np.zeros((original_height, original_width, 4), dtype=np.uint8)

    new_image[paste_y:paste_y + new_height, paste_x:paste_x + new_width] = rescaled_object

    return new_image

class SingleImageDataset(Dataset):
    def __init__(self,
        root_dir: str,
        num_views: int,
        img_wh: Tuple[int, int],
        bg_color: str,
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
            bboxes = []
            scene_list = []
            # Get a list of all files in the directory
            object_names = os.listdir(self.root_dir)
            for object_name in object_names:
                img_path = os.path.join(self.root_dir, object_name,'imgs')
                face_info = self.get_face_info(os.path.dirname(img_path))
                view_names = face_info.keys() # only use the top 10 views
                for view_name in view_names:
                    image_path = os.path.join(img_path, view_name,'rgb.png')
                    if os.path.exists(image_path):
                        file_list.append(image_path)
                        bboxes.append(face_info[view_name]['bbox'])
                    else:
                        raise FileNotFoundError(f"Image file {image_path} not found.")
            # Filter the files that end with .png or .jpg
            self.file_list = file_list
            self.bboxes = bboxes
        else:
            self.file_list = [single_image]
            self.bboxes = [np.array([0, 0, 1, 1])]

 
            
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

    def get_face_info(self, file):
        file_name = os.path.join(file, 'face_info.npy')
        face_info = np.load(file_name, allow_pickle=True).item()
        return face_info
        

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
        bbox = self.bboxes[index]
        if bbox is not None:
            face = self.process_face(file_path, bbox.astype(np.int32), bg_color)
        else:
            face,_ =  self.load_face(file_path, bg_color, return_type='pt')
            face = face.permute(2, 0, 1)
        image, alpha = self.load_image(file_path, bg_color, return_type='pt')
        
        view_path = os.path.dirname(self.file_list[index])
       
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

        

if __name__ == "__main__":
    # pass
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid
    from PIL import ImageDraw, ImageFont
    def draw_text(img, text, pos, color=(128, 128, 128)):
        draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype(size= size)
        font = ImageFont.load_default()
        font = font.font_variant(size=10)
        draw.text(pos, text, color, font=font)
        return img
    random.seed(11)
    test_params = dict(       
        root_dir='../examples/CAPE',
        bg_color='white',
        img_wh=(768, 768),
        prompt_embeds_path='../models/multiview_generator/data/fixed_prompt_embeds_7view',
        num_views=7,
        # crop_size=740,
        margin_size=15,
        smpl_folder='gt_smpl_image',
    )
    train_dataset = SingleImageDataset(**test_params)
    data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    
    for batch in data_loader:
        # batch = train_dataset.__getitem__(1)
        imgs = []
        obj_name = batch['filename'][0]
        imgs_in = batch['imgs_in'][0]
        imgs_smpl_in = batch['smpl_imgs_in'][0]
        alphas_smpl = batch['smpl_alphas'][0]

        img0 = (imgs_in[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img1 = (imgs_smpl_in[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        print(img0.shape, img1.shape)
        smpl_alpha = alphas_smpl[0].permute(1, 2, 0).repeat(1, 1, 3).numpy()
        img0[smpl_alpha > 0.5] = img1[smpl_alpha > 0.5]  
        Image.fromarray(img0).save(f'../../debug/{obj_name}_rgb.png')
        # Image.fromarray(img1).save(f'../../debug/{obj_name}_smpl.png')
    
    exit()
    imgs_vis = torch.cat([imgs_in, imgs_smpl_in], 0)
    img_vis = make_grid(imgs_vis, nrow=4).permute(1, 2,0)
    img_vis = (img_vis.numpy() * 255).astype(np.uint8)
    img_vis = Image.fromarray(img_vis)
    img_vis = draw_text(img_vis, obj_name, (5, 1))
    img_vis = torch.from_numpy(np.array(img_vis)).permute(2, 0, 1) / 255.
    imgs.append(img_vis)
    imgs = torch.stack(imgs, dim=0)
    img_grid = make_grid(imgs, nrow=4, padding=0)
    img_grid = img_grid.permute(1, 2, 0).numpy()
    img_grid = (img_grid * 255).astype(np.uint8)
    img_grid = Image.fromarray(img_grid)
    img_grid.save(f'../../debug/{obj_name}.png')
    print('finished.')

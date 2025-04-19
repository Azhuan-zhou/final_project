import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.multiview_generator.mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline
import torch
from torch.utils.data import DataLoader
from dataset.mv_dataset import SingleImageDataset
from collections import defaultdict
from tqdm.auto import tqdm
from einops import rearrange
import torch.nn.functional as F
from torchvision.utils import save_image


def load_pshuman_pipeline(pretrained_model="stabilityai/stable-diffusion-2-1-unclip",):
    pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(pretrained_model,weight_dtype = torch.float16)
    pipeline.unet.enable_xformers_memory_efficient_attention()
    if torch.cuda.is_available():
        pipeline.to('cuda')
    return pipeline


def prepare_multiview_data(root_dir,batch_size=8,num_workers=1,num_views=5,img_wh=[768, 768],bg_color='white',margin_size=50,single_image=None,filepaths=None, prompt_embeds_path=' models/multiview_generator/data/fixed_prompt_embeds_7view',crop_size=740,smpl_folder='smpl_image_pymaf'):
    dataset = SingleImageDataset(root_dir=root_dir,
                                 num_views=num_views,
                                 img_wh=img_wh,
                                 bg_color=bg_color,
                                 margin_size=margin_size,
                                 single_image=single_image,
                                 prompt_embeds_path=prompt_embeds_path,
                                 crop_size=crop_size,
                                )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    return dataloader




def run_pshuman_pipeline(pipeline,cfg):
    pipeline.set_progress_bar_config(disable=True)
    if cfg.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=pipeline.unet.device).manual_seed(cfg.seed)
        
    dataloader = prepare_multiview_data(cfg.root_dir,
                                       batch_size=cfg.batch_size,
                                       num_views=cfg.num_views,
                                       img_wh=cfg.img_wh,
                                       bg_color=cfg.bg_color,
                                       margin_size=cfg.margin_size,
                                       prompt_embeds_path=cfg.prompt_embeds_path,
                                       crop_size=cfg.crop_size,
                                      )
    
    images_cond = []
    for case_id, batch in tqdm(enumerate(dataloader), desc="Processing", total=len(dataloader)):
        """
        bath
            'imgs_in' (bs, Nv, 3, H, W)
            'normal_prompt_embeddings'
            'color_prompt_embeddings'
            'filename'
        """
        images_cond.append(batch['imgs_in'][:, 0]) # () (bs, 3, H, W) front view
        imgs_in = torch.cat([batch['imgs_in']]*2, dim=0) # (Bs*2, Nv, 3, H, W)
        num_views = imgs_in.shape[1] #
        imgs_in = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W")# (2*bs*Nv, 3, H, W)
        smpl_in = None
        normal_prompt_embeddings, clr_prompt_embeddings = batch['normal_prompt_embeddings'], batch['color_prompt_embeddings'] 
        prompt_embeddings = torch.cat([normal_prompt_embeddings, clr_prompt_embeddings], dim=0)
        prompt_embeddings = rearrange(prompt_embeddings, "B Nv N C -> (B Nv) N C")
        with torch.autocast("cuda"):
            guidance_scale = cfg.validation_guidance_scales
            unet_out = pipeline(
                imgs_in, None, prompt_embeds=prompt_embeddings,
                dino_feature=None, smpl_in=smpl_in,
                generator=generator, guidance_scale=guidance_scale, output_type='pt', num_images_per_prompt=1, 
                **cfg.pipe_validation_kwargs
            )
            
            out = unet_out.images # (2*B*Nv, 3, H, W)
            bsz = out.shape[0] // 2
            normals_pred = out[:bsz] # (B*Nv, 3, H, W)
            images_pred = out[bsz:] # (B*Nv, 3, H, W)
            for i in range(bsz//num_views):
                scene =  batch['filename'][i]
                scene_rgb_dir = os.path.join(scene,'rgb')
                scene_normal_dir = os.path.join(scene,'normal')
                os.makedirs(scene_rgb_dir, exist_ok=True)
                os.makedirs(scene_normal_dir, exist_ok=True)
               

                img_in_ = images_cond[-1][i].to(out.device) # front view(default)
                for j in range(num_views):
                    idx = i*num_views + j
                    normal = normals_pred[idx]
                    if j == 0:
                        color = img_in_ # (3, H, W)
                    else:
                        color = images_pred[idx] # (3, H, W)

                    if j == num_views-1:
                        normal = F.interpolate(normal.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
                     ## save color and normal---------------------
                    normal_filename = f"normals_{j}_masked.png"
                    rgb_filename = f"color_{j}_masked.png"
                    save_image(normal, os.path.join(scene_normal_dir, normal_filename))
                    save_image(color, os.path.join(scene_rgb_dir, rgb_filename))
     
     
def main(cfg):
    pipeline = load_pshuman_pipeline(pretrained_model=cfg.pretrained_model_name_or_path)
    run_pshuman_pipeline(pipeline, cfg)
    print("Done!")
     
# 'front', 'front_right', 'right', 'back', 'left', 'front_left'
if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf
    from configs.misc import load_config, mvConfig
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args, extras = parser.parse_known_args()
     

    # parse YAML config to OmegaConf
    cfg = load_config(args.config, cli_args=extras)
    schema = OmegaConf.structured(mvConfig)
    cfg = OmegaConf.merge(schema, cfg)
    main(cfg)




    
    
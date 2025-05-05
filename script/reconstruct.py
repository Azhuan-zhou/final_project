import PIL.Image as Image
import numpy as np
import os
import torch

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.SiHR import ReconModel
import argparse
from omegaconf import OmegaConf
from configs.misc import load_config, TestConfig
from script.utils import reconstrcut
from dataset.custom_dataset import CustomDataset
import numpy as np
from torch.utils.data import DataLoader
import random
import pickle
from dataset.mesh.load_obj import load_obj
CANONICAL_TEMPLATE = 'data/body_models/smpl_data/smplx_canonical.obj'
WATERTIGHT_TEMPLATE = 'data/body_models/smpl_data/smplx_watertight.pkl'

def move_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        elif isinstance(v, list):
            batch[k] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in v]
        elif k == 'idx':
            continue
    return batch

def prepare_data(cfg,smpl_F):
    dataset = CustomDataset(cfg.data_root,cfg, smpl_F, cfg.device)
    dl =  DataLoader(dataset=dataset,batch_size=1,shuffle=False)
    
    return dl

    
def main(config):
    ckpt = config.ckpt
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
    
    # =================load data=================
    with open(WATERTIGHT_TEMPLATE, 'rb') as f:
        watertight = pickle.load(f)
        smpl_F = watertight['smpl_F']
    can_V, _ = load_obj(CANONICAL_TEMPLATE)
    dl = prepare_data(config,smpl_F)
    model = ReconModel(config, smpl_F,can_V)
    model = model.to(device)
    checkpoint = torch.load(ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    for data in dl:
        data = move_to_device(data, device)
        obj_path = reconstrcut(config,model,data,device)
        print('save obj to ',obj_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,default='configs/test.yaml')
    args, extras = parser.parse_known_args()
     

    # parse YAML config to OmegaConf
    cfg = load_config(args.config, cli_args=extras)
    schema = OmegaConf.structured(TestConfig)
    cfg = OmegaConf.merge(schema, cfg)
    main(cfg)
    
    
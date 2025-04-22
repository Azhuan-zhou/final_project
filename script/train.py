
import os, sys
from models.renderer import Renderer
from dataset.THuman_dataset import THumanReconDataset
import argparse
from omegaconf import OmegaConf
from configs.misc import load_config, mvConfig
from torch.optim.adamw import AdamW
from datetime import datetime
import logging as log
import numpy as np
import torch
import random
import shutil
import tempfile
import pickle
import wandb
from torch.utils.data import DataLoader
from dataset.mesh.load_obj import load_obj
from diffusers.optimization import get_scheduler
from models.evaluator import Evaluator
CANONICAL_TEMPLATE = 'data/smplx_canonical.obj'
WATERTIGHT_TEMPLATE = 'data/smplx_watertight.pkl'

def save_checkpoint(log_dir,epoch,global_step,model,optimizer,full=True,replace=False):
    if replace:
            model_fname = os.path.join(log_dir, f'model.pth')
    else:
        model_fname = os.path.join(log_dir, f'model-{self.epoch:04d}.pth')
        
    state = {
            'epoch': epoch,
            'global_step': global_step,
        }
    if full:
        state['optimizer'] = optimizer.state_dict()
    state['model_state_dict'] = model.state_dict() 
    log.info(f'Saving model checkpoint to: {model_fname}')
    torch.save(state, model_fname)
    
def main(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    log_dir = os.path.join(
            config.save_root,
            config.exp_name,
            f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        )
    device = torch.device(config.device) if torch.cuda.is_available() else torch.device('cpu')
    # =================load data=================
    with open(WATERTIGHT_TEMPLATE, 'rb') as f:
        watertight = pickle.load(f)
    can_V, _ = load_obj(CANONICAL_TEMPLATE)
    # ============================================
    # =================init=================
    model = Renderer(config,watertight,can_V)
    params = [{
        'params':model.parameters(),
        'lr': config.lr_decoder,
    }]
    optimizer = AdamW(params,betas=(config.beta1, config.beta2),
                                    weight_decay=config.weight_decay)
    
    lr_scheduler = get_scheduler(
            config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=config.epochs * 100,
            num_cycles=config.lr_num_cycles,
            power=config.lr_power,
        )

    epoch = config.epoch
    start_epoch = 0
    global_ste = 0
    if config.resume:
        checkpoint = torch.load(config.resume, map_location=device)
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    # ============================================
    # ================data preparation=======================
    dataset = THumanReconDataset(config.data_root, config,mode='train')
    loader = DataLoader(dataset=dataset, 
                        batch_size=config.batch_size, 
                        shuffle=True, 
                        num_workers=config.workers,
                        pin_memory=True)
        
    if config.valid:
        valid_dataset = THumanReconDataset(config.data_root, config,mode='valid')
        valid_loader = DataLoader(valid_dataset, 
                                batch_size=1, 
                                shuffle=False, 
                                num_workers=0,
                                pin_memory=True)
        evaluator = Evaluator(config, watertight, can_V, device,model=model)
    model.train()
    for epoch in range(start_epoch, config.epochs):
        for data in loader:
            optimizer.zero_grad()
            total_loss,reco_loss, rgb_loss, nrm_loss = model(data)
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            if global_step % config.log_every == 0:
                model.log(global_step, epoch,lr_scheduler.get_last_lr()[0])
            global_step += 1
        if epoch % config.save_every == 0:
            save_checkpoint(log_dir,epoch,global_step,model,optimizer,full=True)
        
        if config.valid and epoch % config.valid_every == 0 and epoch != 0:
            torch.cuda.empty_cache()
            model.eval()
            save_path = os.path.join(log_dir, 'meshes')
            os.makedirs(save_path, exist_ok=True)
            for i, data in enumerate(valid_loader):
                if i >= config.num_valid_samples:
                    break
                with torch.no_grad():
                    evaluator.test_reconstruction(
                               data, save_path,global_step,epoch, subdivide=config.subdivide)
                    
            model.train()
            torch.cuda.empty_cache()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args, extras = parser.parse_known_args()
     

    # parse YAML config to OmegaConf
    cfg = load_config(args.config, cli_args=extras)
    schema = OmegaConf.structured(mvConfig)
    cfg = OmegaConf.merge(schema, cfg)
    main(cfg)

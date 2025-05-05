
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.SiHR import ReconModel
from dataset.THuman_dataset import THumanReconDataset
import argparse
from omegaconf import OmegaConf
from configs.misc import load_config, TrainConfig
from torch.optim.adamw import AdamW
from datetime import datetime
import logging as log
import numpy as np
import torch
import random
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset.mesh.load_obj import load_obj
from diffusers.optimization import get_scheduler
from models.evaluator import Evaluator
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

def save_checkpoint(log_dir,epoch,global_step,model,optimizer,full=True,replace=False):
    if replace:
            model_fname = os.path.join(log_dir, f'model.pth')
    else:
        model_fname = os.path.join(log_dir, f'model-{epoch:04d}.pth')
        
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
    os.makedirs(log_dir, exist_ok=True)
    device = torch.device(config.device) if torch.cuda.is_available() else torch.device('cpu')
    n_gpus = torch.cuda.device_count()
    # =================load data=================
    with open(WATERTIGHT_TEMPLATE, 'rb') as f:
        watertight = pickle.load(f)
        smpl_F = watertight['smpl_F']
    can_V, _ = load_obj(CANONICAL_TEMPLATE)
    can_V = can_V
    # ============================================
    # =================init=================
    model = ReconModel(config, smpl_F,can_V)
    if config.distribute:
        assert n_gpus > 1, "Distribute=True but only one GPU is available!"
        device_id = int(cfg.device.split(':')[1])
        model = torch.nn.DataParallel(model, device_ids=[0,1,2], output_device=device_id)
    model = model.to(device)
    model_ref = model.module if isinstance(model, torch.nn.DataParallel) else model

    params_encoder = []
    params_encoder.extend(list(model_ref.feature_extractor.parameters()))

    params_decoder = []
    params_decoder.extend(list(model_ref.geo_model.parameters()))
    params_decoder.extend(list(model_ref.tex_model.parameters()))
    
    params = [
        {
        'params': params_encoder,   
        'lr': config.lr_encoder,
        },
        {
        'params': params_decoder,
        'lr': config.lr_decoder,
        }, 
        ]
    optimizer = AdamW(params,betas=(config.beta1, config.beta2),
                                    weight_decay=config.weight_decay)
    
    

    epochs = config.epochs
    start_epoch = 0
    global_step = 0
    if config.resume:
        checkpoint = torch.load(config.resume, map_location=device)
        start_epoch = checkpoint['epoch']+1
        global_step = checkpoint['global_step']
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        log.info(f"Resume training from epoch {start_epoch} and global step {global_step}")
        #optimizer.load_state_dict(checkpoint['optimizer'])
        
    lr_scheduler = get_scheduler(
            config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=config.epochs * 100,
            num_cycles=config.lr_num_cycles,
            power=config.lr_power,
        )
        
    # ============================================
    start_epoch = 0
    global_step = 0
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
                                num_workers=1,
                                pin_memory=True)
        evaluator_model = model if not isinstance(model, torch.nn.DataParallel) else model.module
        evaluator_model = evaluator_model.to(device)
        evaluator = Evaluator(config, smpl_F, can_V, device,model=evaluator_model)
    model.train()
    for epoch in range(start_epoch, epochs):
        for data in tqdm(loader,desc='Epoch {}/{}:'.format(epoch+1,epochs), unit='batch'):
            data = move_to_device(data, device)
            optimizer.zero_grad()
            total_loss = model(data)
            if cfg.distribute:
                total_loss = total_loss.mean()
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            if global_step % config.log_every == 0:
                if isinstance(model, torch.nn.DataParallel):
                    model.module.log(global_step, epoch, lr_scheduler.get_last_lr()[0])
                else:
                    model.log(global_step, epoch, lr_scheduler.get_last_lr()[0])
            global_step += 1
        if epoch % config.save_every == 0:
            save_checkpoint(log_dir, epoch, global_step, 
                            model.module if isinstance(model, torch.nn.DataParallel) else model,
                            optimizer, full=True)
        
        if config.valid and epoch % config.valid_every == 0 and epoch != 0:
            torch.cuda.empty_cache()
            model.eval()
            save_path = os.path.join(log_dir, 'meshes')
            os.makedirs(save_path, exist_ok=True)
            for i, data in enumerate(valid_loader):
                if i >= config.num_valid_samples:
                    break
                with torch.no_grad():
                    data = move_to_device(data, device)
                    evaluator.test_reconstruction(
                               data, save_path,subdivide=config.subdivide,epoch=epoch)
                    _ = model(data)
            if isinstance(model, torch.nn.DataParallel):
                model.module.log(global_step, epoch, lr_scheduler.get_last_lr()[0])
            else:
                model.log(global_step, epoch, lr_scheduler.get_last_lr()[0])
                    
                    
            model.train()
            torch.cuda.empty_cache()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args, extras = parser.parse_known_args()
     

    # parse YAML config to OmegaConf
    cfg = load_config(args.config, cli_args=extras)
    schema = OmegaConf.structured(TrainConfig)
    cfg = OmegaConf.merge(schema, cfg)
    handlers = [log.StreamHandler(sys.stdout)]
    log.basicConfig(level=cfg.log_level,
                        format='%(asctime)s|%(levelname)8s| %(message)s',
                        handlers=handlers)
    log.info(f'Info: \n{cfg}')
    main(cfg)

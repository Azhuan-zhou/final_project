import os, sys
import argparse
import logging as log
import random
import pickle
import numpy as np
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
from omegaconf import OmegaConf

# ===== 项目依赖 =====
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.SiHR import ReconModel
from dataset.THuman_dataset import THumanReconDataset
from dataset.mesh.load_obj import load_obj
from diffusers.optimization import get_scheduler
from models.evaluator import Evaluator
from configs.misc import load_config, TrainConfig

CANONICAL_TEMPLATE = 'data/smplx_canonical.obj'
WATERTIGHT_TEMPLATE = 'data/smplx_watertight.pkl'


def save_checkpoint(log_dir, epoch, global_step, model, optimizer, full=True, replace=False):
    model_fname = os.path.join(log_dir, f'model.pth' if replace else f'model-{epoch:04d}.pth')
    state = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    }
    if full:
        state['optimizer'] = optimizer.state_dict()
    log.info(f'Saving model checkpoint to: {model_fname}')
    torch.save(state, model_fname)


def main(config, local_rank):
    # ========= 设定随机种子 =========
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # ========= 初始化设备 & 分布式 =========
    if config.distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl')
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    # ========= 加载 watertight 和 canonical mesh =========
    with open(WATERTIGHT_TEMPLATE, 'rb') as f:
        watertight = pickle.load(f)
        smpl_F = watertight['smpl_F'].to(device)
    can_V, _ = load_obj(CANONICAL_TEMPLATE)
    can_V = can_V.to(device)
    # ============================================
    # ========= 初始化模型 =========
    model = ReconModel(config, smpl_F, can_V).to(device)
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    elif torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # ========= 优化器 & 调度器 =========
    optimizer = AdamW([{'params': model.parameters(), 'lr': config.lr_decoder}],
                      betas=(config.beta1, config.beta2),
                      weight_decay=config.weight_decay)
    lr_scheduler = get_scheduler(
        config.lr_scheduler, optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=config.epochs * 100,
        num_cycles=config.lr_num_cycles,
        power=config.lr_power
    )

    # ========= Resume =========
    epoch = config.epochs
    start_epoch = 0
    global_step = 0
    if config.resume:
        checkpoint = torch.load(config.resume, map_location=device)
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # ========= 数据加载器 =========
    dataset = THumanReconDataset(config.data_root, config, mode='train')
    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = DataLoader(dataset, batch_size=config.batch_size,
                            sampler=train_sampler, num_workers=config.workers,
                            pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=True, num_workers=config.workers,
                            pin_memory=True)

    if config.valid:
        valid_dataset = THumanReconDataset(config.data_root, config, mode='valid')
        valid_loader = DataLoader(valid_dataset, batch_size=1,
                                  shuffle=False, num_workers=0, pin_memory=True)
        evaluator = Evaluator(config, watertight, can_V, device, model=model)

    # ========= 日志目录 =========
    log_dir = os.path.join(config.save_root, config.exp_name,
                           f'{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    os.makedirs(log_dir, exist_ok=True)

    # ========= 训练主循环 =========
    model.train()
    for epoch in range(start_epoch, config.epochs):
        if config.distributed:
            train_sampler.set_epoch(epoch)

        for data in loader:
            optimizer.zero_grad()
            total_loss = model(data)
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if global_step % config.log_every == 0:
                model.module.log(global_step, epoch, lr_scheduler.get_last_lr()[0]) \
                    if hasattr(model, 'module') else model.log(global_step, epoch, lr_scheduler.get_last_lr()[0])
            global_step += 1

        if epoch % config.save_every == 0:
            save_checkpoint(log_dir, epoch, global_step, model, optimizer, full=True)

        if config.valid and epoch % config.valid_every == 0 and epoch != 0:
            torch.cuda.empty_cache()
            model.eval()
            save_path = os.path.join(log_dir, 'meshes')
            os.makedirs(save_path, exist_ok=True)
            for i, data in enumerate(valid_loader):
                if i >= config.num_valid_samples:
                    break
                with torch.no_grad():
                    evaluator.test_reconstruction(data, save_path, subdivide=config.subdivide)
                    evaluator.calc_loss(data, global_step, epoch)
            model.train()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--local_rank', type=int, default=0)
    args, extras = parser.parse_known_args()

    cfg = load_config(args.config, cli_args=extras)
    schema = OmegaConf.structured(TrainConfig)
    cfg = OmegaConf.merge(schema, cfg)

    main(cfg, args.local_rank)
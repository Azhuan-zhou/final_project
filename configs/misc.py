import os
from omegaconf import OmegaConf
from packaging import version
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

# ============ Register OmegaConf Recolvers ============= #
OmegaConf.register_new_resolver('calc_exp_lr_decay_rate', lambda factor, n: factor**(1./n))
OmegaConf.register_new_resolver('add', lambda a, b: a + b)
OmegaConf.register_new_resolver('sub', lambda a, b: a - b)
OmegaConf.register_new_resolver('mul', lambda a, b: a * b)
OmegaConf.register_new_resolver('div', lambda a, b: a / b)
OmegaConf.register_new_resolver('idiv', lambda a, b: a // b)
OmegaConf.register_new_resolver('basename', lambda p: os.path.basename(p))
# ======================================================= #


def prompt(question):
    inp = input(f"{question} (y/n)").lower().strip()
    if inp and inp == 'y':
        return True
    if inp and inp == 'n':
        return False
    return prompt(question)


def load_config(*yaml_files, cli_args=[]):
    yaml_confs = [OmegaConf.load(f) for f in yaml_files]
    cli_conf = OmegaConf.from_cli(cli_args)
    conf = OmegaConf.merge(*yaml_confs, cli_conf)
    OmegaConf.resolve(conf)
    return conf


def config_to_primitive(config, resolve=True):
    return OmegaConf.to_container(config, resolve=resolve)


def dump_config(path, config):
    with open(path, 'w') as fp:
        OmegaConf.save(config=config, f=fp)

def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def parse_version(ver):
    return version.parse(ver)



@dataclass
class mvConfig:
    prompt_embeds_path: str
    root_dir: str
    num_views:  int
    bg_color: str
    img_wh:  list
    dataset: str
    crop_size: int
    margin_size: int
    smpl_folder: str
    batch_size: int
    num_workers: int
    seed: int
    validation_guidance_scales: float
    pipe_validation_kwargs: dict
    enable_xformers_memory_efficient_attention: bool
    pretrained_model_name_or_path: str
    
    
    
@dataclass
class TrainConfig:
    #global
    seed: int
    save_root: str
    exp_name: str
    device: str
    dataset: str
    distribute: bool
    #feature_extractor:
    use_global_feature: bool
    use_point_level_feature: bool
    use_pixel_align_feature: bool
    use_trans: bool

    #dataset
    data_root: str
    img_size: int
    num_samples: int
    white_bg: bool
    aug_jitter: bool

    #optimizer
    lr_decoder: float
    lr_encoder: float
    beta1:float
    beta2: float
    weight_decay:float

    #scheduler
    lr_scheduler: str
    lr_warmup_steps: int
    lr_num_cycles: int
    lr_power: float
    max_grad_norm: float

    #train
    epochs: int
    batch_size: int
    workers: int
    save_every: int
    log_every: int
    log_level: int
    resume: Optional[str] 

    #network:
    pos_dim: int
    feat_dim: int
    num_layers: int
    hidden_dim: int
    skip: list
    activation: str
    layer_type: str

    #embedder:
    shape_freq: int
    color_freq: int

    #losses:
    lambda_sdf: float
    lambda_rgb: float
    lambda_nrm: float
    lambda_2D: float
    use_mask: bool
    use_pred_nrm: bool

    #validation:
    valid: bool
    valid_folder: str
    valid_every: int
    subdivide: bool
    grid_size: int
    erode_iter: int
    num_valid_samples: int

    #wandb:
    wandb: bool
    wandb_id: Optional[str] 
    wandb_name: str


class TestConfig:
    #global
    seed: int
    device: str
    save_root: str
    exp_name: str
    dataset: str
    ckpt: str
    
    # test
    subdivide: bool
    save_uv: bool
    clean_mesh_flag: bool
    mcube_res: int
    
    #dataset
    data_root: str
    smpl_root: str
    img_size: int
    num_samples: int
    white_bg: bool
    aug_jitter: bool
    
    #feature_extractor:
    use_trans: bool
    
    #network:
    pos_dim: int
    feat_dim: int
    num_layers: int
    hidden_dim: int
    skip: list
    activation: str
    layer_type: str
    #embedder:
    shape_freq: int
    color_freq: int
    
    #validation:
    grid_size: int




import torch
import os.path as osp
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from script.utils import clean_mesh,reconstrcut, VF2Mesh,calculate_chamfer_p2s,calculate_normal_consist, accumulate
from models.Render import Render
import time
import trimesh
import random
from termcolor import colored
from models.SiHR import ReconModel
from dataset.mesh.load_obj import load_obj
import pickle
from dataset.THuman_dataset import THumanReconDataset
from dataset.CAPE_dataset import CapeReconDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
torch.backends.cudnn.benchmark = True
CANONICAL_TEMPLATE = 'data/body_models/smpl_data/smplx_canonical.obj'
SMPL_CANONICAL_TEMPLATE = 'data/body_models/smpl_data/smpl_canonical.obj'
WATERTIGHT_TEMPLATE = 'data/body_models/smpl_data/smplx_watertight.pkl'
SMPL_NATURAL = 'data/body_models/smpl/SMPL_NEUTRAL.pkl'

def move_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        elif isinstance(v, list):
            batch[k] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in v]
        elif k == 'idx':
            continue
    return batch
        
def test_step(cfg,model,batch,device):
    
        resolutions = (np.logspace(
            start=5,
            stop=np.log2(cfg.mcube_res),
            base=2,
            num=int(np.log2(cfg.mcube_res) - 4),
            endpoint=True,
        ) + 1.0)
        resolutions = resolutions.astype(np.int16).tolist()
        # export paths
        mesh_name = batch["idx"][0]
        
       


        start_time=time.time()
        with torch.no_grad():
            pre_obj_path = reconstrcut(cfg,model,batch,device)
        end_time=time.time()
        mesh_pred = trimesh.load(pre_obj_path)
        mesh_gt = trimesh.load(os.path.join(cfg.GT_root, '{}.obj'.format(mesh_name)))
        verts_pr, faces_pr = mesh_pred.vertices, mesh_pred.faces
        verts_gt, faces_gt = mesh_gt.vertices, mesh_gt.faces

        if cfg.clean_mesh_flag:
            verts_pr, faces_pr = clean_mesh(verts_pr, faces_pr,device)
        
        icp = trimesh.registration.icp(verts_pr, verts_gt)

        mesh_pred_icp = trimesh.Trimesh(vertices=icp[1], faces=faces_pr, process=False)
        
        verts_pred_icp, faces_pred_icp = mesh_pred_icp.vertices, mesh_pred_icp.faces
        verts_pred_icp = torch.tensor(verts_pred_icp).float().to(device)
        faces_pred_icp = torch.tensor(faces_pred_icp).long().to(device)
        verts_gt = torch.tensor(verts_gt).float().to(device)    
        faces_gt = torch.tensor(faces_gt).long().to(device)
        src_mesh = VF2Mesh(verts_pred_icp, faces_pred_icp )
        tgt_mesh = VF2Mesh(verts_gt, faces_gt)
        
        chamfer, p2s = calculate_chamfer_p2s(tgt_mesh, src_mesh, num_samples=1000)
        render = Render(size=512, device=device)
        export_dir = osp.join(cfg.save_root,cfg.dataset ,cfg.exp_name, mesh_name)
        os.makedirs(export_dir, exist_ok=True)
        nrm_path =  osp.join(export_dir, f"nc.png")
        normal_consist = calculate_normal_consist(tgt_mesh,src_mesh,nrm_path ,render)
        
        execution_time=end_time-start_time
        test_log = {"chamfer": chamfer, "p2s": p2s, "NC": normal_consist,"execution_time":execution_time}

        return test_log
    
def test_epoch_end(cfg,outputs):
        chamfer=[]
        p2s=[]
        NC=[] 
        for item in outputs:
            chamfer.append(item["chamfer"])
            p2s.append(item["p2s"])
            NC.append(item["NC"])
        chamfer=torch.tensor(chamfer)
        p2s=torch.tensor(p2s)
        NC=torch.tensor(NC)
        print('chamfer: ', torch.mean(chamfer))
        print('p2s: ', torch.mean(p2s))
        print('NC: ', torch.mean(NC))

        print(colored(cfg.dataset, "green"))
        if cfg.dataset  == 'cape':
            split = {
                "cape-easy": (0, 50),
                "cape-hard": (50, 150)
            }
        elif cfg.dataset  == 'THuman':
            split = {
                "Thuman2.0": (0,21)
            }
        else:
            raise ValueError("Unknown dataset type")
        accu_outputs = accumulate(
            outputs,
            rot_num=1,
            split=split,
        )
        export_folder = osp.join(cfg.save_root, cfg.dataset ,cfg.exp_name)
        np.save(
            osp.join(export_folder, "test_results.npy"),
            accu_outputs,
            allow_pickle=True,
        )

        return accu_outputs
    
def main(config):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
   
    if config.dataset == 'THuman':
        with open(WATERTIGHT_TEMPLATE, 'rb') as f:
            smpl_model = pickle.load(f)
            smpl_F = smpl_model['smpl_F']
        can_V, _ = load_obj(CANONICAL_TEMPLATE)
        test_dataset = THumanReconDataset(config.data_root, config,mode='test')
    elif config.dataset == 'cape':
        with open(SMPL_NATURAL, 'rb') as f:
            smpl_model = pickle.load(f, encoding='latin1')
            smpl_F = smpl_model['f']
            smpl_F = torch.as_tensor(smpl_F.astype(np.int32)).long()
        can_V, _ = load_obj(SMPL_CANONICAL_TEMPLATE)
        test_dataset = CapeReconDataset(config.data_root, config, smpl_F,device)
        
        
    
    model = ReconModel(config,smpl_F,can_V)
    model = model.to(device)
    checkpoint = torch.load(config.ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loader = DataLoader(test_dataset, 
                                batch_size=1, 
                                shuffle=False, 
                                num_workers=0,
                                pin_memory=True)
    outputs = []
    for data in tqdm(test_loader,desc='Testing'):
        data = move_to_device(data, device)
        test_log = test_step(config,model,data,device)
        outputs.append(test_log)
    test_epoch_end(cfg,outputs)
        
        
    
    
    
if __name__ == "__main__":
    from omegaconf import OmegaConf
    from configs.misc import load_config, TestConfig
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args, extras = parser.parse_known_args()
     

    # parse YAML config to OmegaConf
    cfg = load_config(args.config, cli_args=extras)
    schema = OmegaConf.structured(TestConfig)
    cfg = OmegaConf.merge(schema, cfg)
    main(cfg)

    
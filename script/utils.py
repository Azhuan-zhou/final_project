import random
import torch
from models.evaluator import Evaluator
import trimesh
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss.point_mesh_distance import _PointFaceDistance
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.structures import Meshes
from dataset.mesh.load_obj import load_obj
from torchvision.utils import make_grid
import pickle
import os
import numpy as np
from PIL import Image
from termcolor import colored
####################################################
CANONICAL_TEMPLATE = 'data/body_models/smpl_data/smplx_canonical.obj'
WATERTIGHT_TEMPLATE = 'data/body_models/smpl_data/smplx_watertight.pkl'
####################################################
with open(WATERTIGHT_TEMPLATE, 'rb') as f:
        watertight = pickle.load(f)
can_V, _ = load_obj(CANONICAL_TEMPLATE)


def accumulate(outputs, rot_num, split):

    hparam_log_dict = {}

    metrics = outputs[0].keys()
    datasets = split.keys()

    for dataset in datasets:
        for metric in metrics:
            keyword = f"{dataset}-{metric}"
            if keyword not in hparam_log_dict.keys():
                hparam_log_dict[keyword] = 0
            for idx in range(split[dataset][0] * rot_num,
                             split[dataset][1] * rot_num):
                hparam_log_dict[keyword] += outputs[idx][metric]
            hparam_log_dict[keyword] /= (split[dataset][1] -
                                         split[dataset][0]) * rot_num

    print(colored(hparam_log_dict, "green"))

    return hparam_log_dict

def VF2Mesh(verts, faces, vertex_texture = None):
    device = verts.device
    if not torch.is_tensor(verts):
        verts = torch.tensor(verts)
    if not torch.is_tensor(faces):
        faces = torch.tensor(faces)
    if verts.ndimension() == 2:
        verts = verts.unsqueeze(0).float()
    if faces.ndimension() == 2:
        faces = faces.unsqueeze(0).long()
    verts = verts.to(device)
    faces = faces.to(device)
    if vertex_texture is not None:
        vertex_texture = vertex_texture.to(device)
    mesh = Meshes(verts, faces).to(device)
    if vertex_texture is None:
        mesh.textures = TexturesVertex(
            verts_features=(mesh.verts_normals_padded() + 1.0) * 0.5)#modify
    else:    
        mesh.textures = TexturesVertex(
            verts_features = vertex_texture.unsqueeze(0))#modify
    return mesh
    
def point_mesh_distance(meshes, pcls):

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()

    # point to face distance: shape (P,)
    point_to_face = _PointFaceDistance.apply(points, points_first_idx, tris,
                                             tris_first_idx, max_points, 5e-3)

    # weight each example by the inverse of number of points in the example
    point_to_cloud_idx = pcls.packed_to_cloud_idx()  # (sum(P_i),)
    num_points_per_cloud = pcls.num_points_per_cloud()  # (N,)
    weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
    weights_p = 1.0 / weights_p.float()
    point_to_face = torch.sqrt(point_to_face) * weights_p
    point_dist = point_to_face.sum() / N

    return point_dist

def calculate_chamfer_p2s(tgt_mesh,src_mesh, num_samples=1000):

        tgt_points = Pointclouds(
            sample_points_from_meshes(tgt_mesh, num_samples))
        src_points = Pointclouds(
            sample_points_from_meshes(src_mesh, num_samples))
        p2s_dist = point_mesh_distance(src_mesh, tgt_points) * 100.0
        chamfer_dist = (point_mesh_distance(tgt_mesh, src_points) * 100.0
                        + p2s_dist) * 0.5

        return chamfer_dist, p2s_dist
    

def calculate_normal_consist(tgt_mesh,src_mesh,normal_path,render):

        src_normal_imgs = render.get_rgb_image(src_mesh,cam_ids=[ 0,1,2, 3],
                                                    bg='black')

        tgt_normal_imgs = render.get_rgb_image(tgt_mesh,cam_ids=[0,1,2, 3],
                                                    bg='black')
        
        src_normal_arr = make_grid(torch.cat(src_normal_imgs, dim=0), nrow=4,padding=0)  # [0,1]
        tgt_normal_arr = make_grid(torch.cat(tgt_normal_imgs, dim=0), nrow=4,padding=0)  # [0,1]
        src_norm = torch.norm(src_normal_arr, dim=0, keepdim=True)
        tgt_norm = torch.norm(tgt_normal_arr, dim=0, keepdim=True)

        src_norm[src_norm == 0.0] = 1.0
        tgt_norm[tgt_norm == 0.0] = 1.0

        src_normal_arr /= src_norm
        tgt_normal_arr /= tgt_norm

        src_normal_arr = (src_normal_arr + 1.0) * 0.5
        tgt_normal_arr = (tgt_normal_arr + 1.0) * 0.5
        error = ((
                (src_normal_arr - tgt_normal_arr)**2).sum(dim=0).mean()) * 4
        #print('normal error:', error)

        normal_img = Image.fromarray(
                (torch.cat([src_normal_arr, tgt_normal_arr], dim=1).permute(
                    1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8))
        normal_img.save(normal_path)
        
        error_list = []
        if len(src_normal_imgs) > 4:
            for i in range(len(src_normal_imgs)):
                src_normal_arr = src_normal_imgs[i]  # Get each source normal image
                tgt_normal_arr = tgt_normal_imgs[i]  # Get corresponding target normal image

                src_norm = torch.norm(src_normal_arr, dim=0, keepdim=True)
                tgt_norm = torch.norm(tgt_normal_arr, dim=0, keepdim=True)

                src_norm[src_norm == 0.0] = 1.0
                tgt_norm[tgt_norm == 0.0] = 1.0

                src_normal_arr /= src_norm
                tgt_normal_arr /= tgt_norm

                src_normal_arr = (src_normal_arr + 1.0) * 0.5
                tgt_normal_arr = (tgt_normal_arr + 1.0) * 0.5

                error = ((src_normal_arr - tgt_normal_arr) ** 2).sum(dim=0).mean() * 4.0
                error_list.append(error)

               
            return error_list
        else:
            src_normal_arr = make_grid(torch.cat(src_normal_imgs, dim=0), nrow=4,padding=0)  # [0,1]
            tgt_normal_arr = make_grid(torch.cat(tgt_normal_imgs, dim=0), nrow=4,padding=0)  # [0,1]
            src_norm = torch.norm(src_normal_arr, dim=0, keepdim=True)
            tgt_norm = torch.norm(tgt_normal_arr, dim=0, keepdim=True)

            src_norm[src_norm == 0.0] = 1.0
            tgt_norm[tgt_norm == 0.0] = 1.0

            src_normal_arr /= src_norm
            tgt_normal_arr /= tgt_norm

            # sim_mask = self.get_laplacian_2d(tgt_normal_arr).to(self.device)

            src_normal_arr = (src_normal_arr + 1.0) * 0.5
            tgt_normal_arr = (tgt_normal_arr + 1.0) * 0.5

            error = ((
                (src_normal_arr - tgt_normal_arr)**2).sum(dim=0).mean()) * 4
            #print('normal error:', error)
            return error
        
def clean_mesh(verts, faces,device):

    mesh_lst = trimesh.Trimesh(verts,
                               faces)
    mesh_lst = mesh_lst.split(only_watertight=False)
    comp_num = [mesh.vertices.shape[0] for mesh in mesh_lst]
    mesh_clean = mesh_lst[comp_num.index(max(comp_num))]

    final_verts = torch.as_tensor(mesh_clean.vertices).float().cpu().numpy()
    final_faces = torch.as_tensor(mesh_clean.faces).int().cpu().numpy()

    return final_verts, final_faces


def reconstrcut(config,model,data,device):
    # Set random seed.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    evaluator = Evaluator(config, watertight, can_V,device,model)
    save_path = os.path.join(config.save_root,config.dataset ,config.exp_name,'meshes')
    if os.path.exists(save_path):
        fname = data['idx'][0]
        obj_path = os.path.join(save_path, '%s_reco.obj' % (fname))
        if os.path.exists(obj_path):
            print('Mesh already exists, skip reconstruction')
            return obj_path
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        obj_path = evaluator.test_reconstruction(data, save_path, subdivide=config.subdivide, save_uv=config.save_uv,chunk_size=6e5)
    return obj_path




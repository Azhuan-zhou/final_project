"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""

import os
import math
import torch
import nvdiffrast
import pickle

import kaolin as kal
import PIL.Image as Image
import numpy as np

from dataset.mesh import load_obj, point_sample, closest_tex, per_face_normals, sample_tex
from dataset.load_smplx_json import load_json
import pdb
###################################################################
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
glctx = nvdiffrast.torch.RasterizeCudaContext(device=device)

UV_TEMPLATE = 'data/body_models/smplx_uv.obj'
WATERTIGHT_TEMPLATE = 'data/body_models/smplx_watertight.pkl'

###################################################################



def rasterize(camera, V, F, args):

    vertices_camera = camera.extrinsics.transform(V)
    face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(
                            vertices_camera, F)
    face_normals_z = kal.ops.mesh.face_normals(face_vertices_camera,unit=True)[..., -1:].contiguous()
    proj = camera.projection_matrix()[0:1]
    homogeneous_vecs = kal.render.camera.up_to_homogeneous(
        vertices_camera
    )[..., None]
    vertices_clip = (proj @ homogeneous_vecs).squeeze(-1)
    rast = nvdiffrast.torch.rasterize(
        glctx, vertices_clip, F.int(),
        (args.size, args.size), grad_db=False
    )
    rast0 = torch.flip(rast[0], dims=(1,))
    hard_mask = rast0[:, :, :, -1:] != 0

    return rast0, hard_mask



def init_camera(args):

    def cartesian_to_spherical(xyz):
        radius = torch.sqrt(torch.sum(xyz**2, axis=1))
        xz = torch.sqrt(xyz[:,0]**2 + xyz[:,2]**2)
        elevation = torch.atan2(xyz[:,1], xz) 
        azimuth = torch.atan2(xyz[:,0], xyz[:,2])

        return torch.rad2deg(elevation), torch.rad2deg(azimuth), radius


    look_at = torch.zeros( (args.nviews, 3), dtype=torch.float32)
    camera_up_direction = torch.tensor( [[0, 1, 0]], dtype=torch.float32).repeat(args.nviews, 1,)

    if args.camera_sampling == 'uniform':
        angle = torch.linspace(0, 2*np.pi, args.nviews+1)[:-1]
        camera_position = torch.stack( (3*torch.sin(angle), torch.zeros_like(angle), 3*torch.cos(angle)), dim=1)
    elif args.camera_sampling == 'random':
        azimuth_range = [-180, 180]
        elevation_range = [-10, 10]
        azimuth_deg = torch.rand(args.nviews // 2) * (azimuth_range[1] - azimuth_range[0]) + azimuth_range[0]
        azimuth = azimuth_deg * math.pi / 180

        elevation_deg = torch.rand(args.nviews // 2) * (elevation_range[1] - elevation_range[0]) + elevation_range[0]
        elevation = elevation_deg * math.pi / 180

        radius = 3.0

        cam_front = torch.stack(
            [
                radius * torch.cos(elevation) * torch.sin(azimuth),
                radius * torch.sin(elevation),
                radius * torch.cos(elevation) * torch.cos(azimuth),

            ]
            , dim=-1
        )
        new_azimuth = torch.where(azimuth > 0., -math.pi + azimuth, math.pi + azimuth)
        cam_back = torch.stack(
                [
                    radius * torch.cos(-elevation) * torch.sin(new_azimuth),
                    radius * torch.sin(-elevation),
                    radius * torch.cos(-elevation) * torch.cos(new_azimuth),
                ]
                , dim = -1
        )

        camera_position = torch.cat([cam_front, cam_back], dim=0)
    else:
        raise NotImplementedError

    if args.camera_mode == 'orth':
        camera = kal.render.camera.Camera.from_args(eye=camera_position,
                                         at=look_at,
                                         up=camera_up_direction,
                                         width=args.size, height=args.size,
                                         near=-512, far=512,
                                        fov_distance=1.0, device=device)
    
    elif args.camera_mode == 'persp':
        camera = kal.render.camera.Camera.from_args(eye=camera_position,
                                         at=look_at,
                                         up=camera_up_direction,
                                         fov=45 * np.pi / 180,
                                         width=args.size, height=args.size,
                                         near=0.01, far=10,
                                         device=device)
    else:
        raise NotImplementedError

    cam_pos = camera.extrinsics.cam_pos()
    elevation, azimuth, radius = cartesian_to_spherical(cam_pos)

    camera_dict = {}
    camera_dict['cam_pos'] = cam_pos.cpu().numpy()
    camera_dict['elevation'] = elevation.cpu().numpy()
    camera_dict['azimuth'] = azimuth.cpu().numpy()
    camera_dict['radius'] = radius.cpu().numpy()


    return camera, camera_dict

def render_visiblity(camera, smpl_file, args):

    smpl_V, smpl_F, joint_3d = load_json(smpl_file, device=device)
    
    with open(WATERTIGHT_TEMPLATE, 'rb') as f:
        smpl_mesh = pickle.load(f)

    F = smpl_mesh['smpl_F'].to(device)

    rast0, _ = rasterize(camera, smpl_V, F, args)

    face_idx = (rast0[..., -1:].long() - 1).contiguous()
    # assign visibility to 1 if face_idx >= 0
    vv = []
    for i in range(rast0.shape[0]):
        vis = torch.zeros((F.shape[0],), dtype=torch.bool, device=device)
        for f in range(F.shape[0]):
            vis[f] = 1 if torch.any(face_idx[i] == f) else 0
        vv.append(vis)
    vis = torch.stack(vv, dim=0)

    return smpl_V, joint_3d, vis

def sample_points(obj_file, args):

    out = load_obj(obj_file, load_materials=True)
    mesh_V, mesh_F, texv, texf, mats = out    #print(mesh.materials[0])


    pts = point_sample(mesh_V.to(device),
                      mesh_F.to(device),
                      ['rand', 'near', 'near', 'trace'],
                      args.nsamples, variance = 0.01)

    # Randomly shuffle the points as we cannot do this in the h5f file
    idx = torch.randperm(pts.shape[0])
    pts = pts[idx]

    rgb, nrm, d = closest_tex(mesh_V.to(device), mesh_F.to(device),
                                texv.to(device), texf.to(device), mats, pts)

    point_cloud = {
        'xyz': pts.cpu().numpy(),
        'rgb': rgb.cpu().numpy(),
        'nrm': nrm.cpu().numpy(),
        'd': d.cpu().numpy(),
    }
    return point_cloud



def render_images(camera, obj_file, args):

    # Render RGBA and normal map from the mesh

    out = load_obj(obj_file, load_materials=True)
    mesh_V, mesh_F, texv, texf, mats = out

    FN = per_face_normals(mesh_V, mesh_F).to(device)

    rast1, _ = rasterize(camera, mesh_V.to(device), mesh_F.to(device), args)

    face_idx = (rast1[..., -1].long() - 1).contiguous()

    uv_map = nvdiffrast.torch.interpolate(
       texv.to(device), rast1, texf[...,:3].int().to(device)
    )[0] % 1.

    nrm_map = FN[face_idx.view(-1)].view(args.nviews, args.size, args.size, 3)

    TM = torch.zeros((args.nviews, args.size, args.size, 1), dtype=torch.long, device=device)
    rgb = sample_tex(uv_map.view(-1, 2), TM.view(-1), mats).view(args.nviews, args.size, args.size, 3)
    mask = (face_idx != -1)

    return rgb.data.cpu(), nrm_map.data.cpu(), mask.data.cpu()


def generate_data(local_path, args):
    obj_name = local_path.split('/')[-1]
    img_save_dir = os.path.join(args.output_path, obj_name,'imgs')
    os.makedirs(img_save_dir, exist_ok=True)
    npy_save_dir = os.path.join(args.output_path, obj_name,'params')
    os.makedirs(npy_save_dir, exist_ok=True)
    point_save_dir = os.path.join(args.output_path, obj_name,'points')
    os.makedirs(point_save_dir, exist_ok=True)
    # Load the 3D scan and SMPL-X registration
    obj_file = [os.path.join(local_path, f) for f in os.listdir(local_path) if f.endswith(('ply','obj')) 
                 and not f.endswith(('_smpl.obj', '_smplx.obj')) ][0]
    smpl_file = [os.path.join(local_path, f) for f in os.listdir(local_path) if f.endswith('json')][0]

    # Initialize the camera
    cameras, camera_dict = init_camera(args)
    cam_eva = camera_dict['elevation'].reshape(-1, 1)
    cam_azh = camera_dict['azimuth'].reshape(-1, 1)
    cam_rad = camera_dict['radius'].reshape(-1, 1)
    cam_pos = camera_dict['cam_pos'].reshape(-1, 3)
    R = camera_dict['R']
    T = camera_dict['T']
    K = camera_dict['K']

    camera_dict = {
        'cam_eva': cam_eva,
        'cam_azh': cam_azh, 
        'cam_rad': cam_rad,
        'cam_pos': cam_pos,
        'R': R,
        'T': T,
        'K': K,
    }
    np.save(os.path.join(npy_save_dir, 'camera.npy'), camera_dict)
    point_cloud = sample_points(obj_file, args)
    pts = point_cloud['xyz']
    rgb = point_cloud['rgb']
    nrm = point_cloud['nrm']
    d = point_cloud['d']
    point_dict = {
        'xyz': pts,
        'rgb': rgb,
        'nrm': nrm,
        'd': d,
    }
    np.save(os.path.join(point_save_dir, 'point.npy'), point_dict)
    # Get SMPL-X vertices, joints and visibility labels from the camera
    
    V, joint_3d, vis= render_visiblity(cameras, smpl_file, args)
    smpl_v = V.cpu().numpy()
    joint_3d = joint_3d.cpu().numpy()
    vis = vis.cpu().numpy()
    smpl_dict = {
        'vertices': smpl_v,
        'joints': joint_3d,
        'vis': vis,
    }
    np.save(os.path.join(npy_save_dir, 'smpl.npy'), smpl_dict)

    # Render the images from the mesh and SMPL-X
    #pdb.set_trace()
    rgb,nrm, mask = render_images(cameras, obj_file, smpl_file, args)
    
    
    for i in range(args.nviews):
        os.makedirs(os.path.join(img_save_dir, f'view{i:02d}'), exist_ok=True)
        # Save RGB image as binary png file
        img = torch.zeros((args.size, args.size, 4))
        img[...,:3] = rgb[i]
        img[...,3] = mask[i]
        rgba = Image.fromarray((255 * img).numpy().astype(np.uint8), mode='RGBA')
        rgba.save(os.path.join(img_save_dir, f'view{i:02d}','rgb.png'))
        
        # Save normal map as binary png file
        img = torch.zeros((args.size, args.size, 4))
        img[...,:3] = (nrm[i] * 0.5 + 0.5)
        img[...,3] = mask[i]
        nrm_img = Image.fromarray((255 * img).numpy().astype(np.uint8), mode='RGBA')
        nrm_img.save(os.path.join(img_save_dir, f'view{i:02d}','nrm.png'))
            
       

       
       
            
"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
import torch
import torch.nn as nn
import time
import trimesh
import os
import logging as log
import cv2
import numpy as np
import nvdiffrast.torch as dr
from PIL import Image
from kaolin.ops.conversions import voxelgrids_to_trianglemeshes
from kaolin.ops.mesh import subdivide_trianglemesh
import torch.nn.functional as F
from .networks.normal_predictor import define_G
from .geo_model import GeoModel
from .tex_model import TexModel
from .SMPL_query import SMPL_query
from tqdm import tqdm
class Evaluator(nn.Module):

    def __init__(self, config, smpl_F, can_V, device,model):

        super().__init__()

        # Set device to use
        self.device = device
        self.glctx = dr.RasterizeCudaContext(device=self.device)

        #device_name = torch.cuda.get_device_name(device=self.device)
        #log.info(f'Using {device_name} with CUDA v{torch.version.cuda}')

        self.cfg = config

        # create marching cube grid
        self.res = self.cfg.grid_size
        window_x = torch.linspace(-1., 1., steps=self.res, device=device)
        window_y = torch.linspace(-1., 1., steps=self.res, device=device)
        window_z = torch.linspace(-1., 1., steps=self.res, device=device)

        self.coord = torch.stack(torch.meshgrid(window_x, window_y, window_z, indexing='ij')).permute(
                                1, 2, 3, 0).reshape(1, -1, 3).contiguous()

        self.smpl_query = SMPL_query(smpl_F, can_V)

        self.model = model
        self.sdf = None
        self.nrm = None
        self.rgb = None
        
    def _repair_mesh(self, mesh):

        # remove disconnect par of mesh
        connected_comp = mesh.split(only_watertight=False)
        max_area = 0
        max_comp = None
        for comp in connected_comp:
            if comp.area > max_area:
                max_area = comp.area
                max_comp = comp
        mesh = max_comp
            
        trimesh.repair.fix_inversion(mesh)

        return mesh

    def _uv_padding(self, image, hole_mask):
        inpaint_image = (
            cv2.inpaint(
                (image.detach().cpu().numpy() * 255).astype(np.uint8),
                (hole_mask.detach().cpu().numpy() * 255).astype(np.uint8),
                4,
                cv2.INPAINT_TELEA,
                )
            )
        return inpaint_image

    def test_reconstruction(self, data, save_path, subdivide=True, chunk_size=5e5, flip=False, save_uv=False,epoch=None):
        # batch_size must be 1
        assert len(data['idx']) == 1, "Batch size must be 1 for reconstruction"
        
        fname = data['idx'][0]
        log.info(f"Reconstructing mesh for {fname}...")

        start = time.time()
        # first estimate the sdf values
        _points = torch.split(self.coord, int(chunk_size), dim=1)
        voxels = []

        with torch.no_grad():
            for _p in tqdm(_points,desc='Marching cube', unit='chunk'):
                _p = _p.to(self.device)
                pred_sdf,pred_nrm,_ = self.model.forward_3D(data, _p,geo=True,tex=False)
                voxels.append(pred_sdf)

        voxels = torch.cat(voxels, dim=1)[..., 0]
        voxels = voxels.reshape(1, self.res, self.res, self.res)
        
        vertices, faces = voxelgrids_to_trianglemeshes(voxels, iso_value=0.)
        vertices = ((vertices[0].reshape(1, -1, 3) - 0.5) / (self.res/2)) - 1.0
        faces = faces[0]

        if subdivide:
            vertices, faces = subdivide_trianglemesh(vertices, faces, iterations=1)

        # Next estimate the texture rgb values on the surface
        if save_uv:
            
            d = trimesh.Trimesh(vertices=vertices[0].cpu().detach().numpy(), 
                faces=faces.cpu().detach().numpy(), 
                process=False)
            d = self._repair_mesh(d)

            import xatlas
            vmap, uv_faces, uvs = xatlas.parametrize(d.vertices, d.faces)
            faces_tensor = torch.from_numpy(uv_faces.astype(np.int32)).to(self.device)

            uv_clip = torch.from_numpy(uvs) * torch.tensor([2.0, -2.0]) + torch.tensor([-1.0, 1.0])
            # pad to four component coordinate
            uv_clip4 = torch.cat(
                (
                    uv_clip,
                    torch.zeros_like(uv_clip[..., 0:1]),
                    torch.ones_like(uv_clip[..., 0:1]),
                ),
                dim=-1,
            ).to(self.device)
            # rasterize
            rast, _ = dr.rasterize(self.glctx, uv_clip4[None,...], faces_tensor, (1024, 1024), grad_db=True)
            #rast0 = torch.flip(rast[0], dims=(1,))
            rast0 = rast[0]
            hole_mask = ~(rast0[:, :, 3] > 0)
            gb_pos, _ = dr.interpolate(torch.from_numpy(d.vertices).to(self.device).float(), rast, torch.from_numpy(d.faces).to(self.device).int())
            gb_pos = gb_pos[0].view(1,-1, 3)
            _points = torch.split(gb_pos, int(chunk_size), dim=1)

            pred_rgb = []
            with torch.no_grad():
                for _p in tqdm(_points,desc='Texture prediction', unit='chunk'):
                    _p = _p.to(self.device)
                    _,_,output = self.model.forward_3D(data, _p, geo=False, tex=True)
                    pred_rgb.append(output)

            pred_rgb = torch.cat(pred_rgb, dim=1).view(1024, 1024, 3)

            pad_rgb = self._uv_padding(pred_rgb, hole_mask)

            texture_map = Image.fromarray(pad_rgb)

            h = trimesh.Trimesh(
                vertices=d.vertices[vmap],
                faces=uv_faces,
                visual=trimesh.visual.TextureVisuals(uv=uvs, image=texture_map),
                process=False,
            )

            if flip: # flip to the gradio coordinate system
                h.apply_transform( [[-1, 0, 0, 0],
                                [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]] )

            h.visual.material.name = fname
            obj_path = os.path.join(save_path, '%s_reco.obj' % (fname))
            h.export(obj_path, mtl_name=fname+'.mtl')

            with open(os.path.join(save_path, fname+'.mtl'), 'w') as f:
                f.write('newmtl {}\n'.format(fname))
                f.write('map_Kd {}.png\n'.format(fname))

        else:
            _points = torch.split(vertices, int(chunk_size), dim=1)
            pred_rgb = []
            with torch.no_grad():
                for _p in tqdm(_points,desc='Texture prediction', unit='chunk'):
                    _,_,output = self.model.forward_3D(data, _p, geo=False, tex=True)
                    pred_rgb.append(output)

            pred_rgb = torch.cat(pred_rgb, dim=1)
        
            h = trimesh.Trimesh(vertices=vertices[0].cpu().detach().numpy(), 
                faces=faces.cpu().detach().numpy(), 
                vertex_colors=pred_rgb[0].cpu().detach().numpy(),
                process=False)

            if flip: # flip to the gradio coordinate system
                h.apply_transform( [[-1, 0, 0, 0],
                                [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]] )


            h = self._repair_mesh(h)
            if epoch is not None:
                obj_path = os.path.join(save_path, '%s_reco_%04d.obj' % (fname, epoch))
            else:
                obj_path = os.path.join(save_path, '%s_reco.obj' % (fname))
            h.export(obj_path)
        end = time.time()
        log.info(f"Reconstruction finished in {end-start} seconds.")
        
        return obj_path
       

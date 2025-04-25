from pytorch3d.renderer import (
    BlendParams,
    blending,
    look_at_view_transform,
    FoVOrthographicCameras,
    PointLights,
    RasterizationSettings,
    PointsRasterizationSettings,
    PointsRenderer,
    AlphaCompositor,
    PointsRasterizer,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import rasterize_meshes


from models.render_utils import Pytorch3dRasterizer
import torch
import numpy as np

import cv2
def get_visibility_color(xy, z, faces):
    """get the visibility of vertices

    Args:
        xy (torch.tensor): [N,2]
        z (torch.tensor): [N,1]
        faces (torch.tensor): [N,3]
        size (int): resolution of rendered image
    """

    xyz = torch.cat((xy, -z), dim=1)
    xyz = (xyz + 1.0) / 2.0
    faces = faces.long()

    rasterizer = Pytorch3dRasterizer(image_size=2**12)
    meshes_screen = Meshes(verts=xyz[None, ...], faces=faces[None, ...])
    raster_settings = rasterizer.raster_settings

    pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
        meshes_screen,
        image_size=raster_settings.image_size,
        blur_radius=raster_settings.blur_radius,
        faces_per_pixel=raster_settings.faces_per_pixel,
        bin_size=raster_settings.bin_size,
        max_faces_per_bin=raster_settings.max_faces_per_bin,
        perspective_correct=raster_settings.perspective_correct,
        cull_backfaces=raster_settings.cull_backfaces,
    )

    vis_vertices_id = torch.unique(faces[torch.unique(pix_to_face), :])
    vis_mask = torch.zeros(size=(z.shape[0], 1))
    vis_mask[vis_vertices_id] = 1.0

    # 新增的部分: 检测边缘像素
    edge_mask = torch.zeros_like(pix_to_face)
    offset=1
    for i in range(-1-offset, 2+offset):
        for j in range(-1-offset, 2+offset):
            if i == 0 and j == 0:
                continue
            shifted = torch.roll(pix_to_face, shifts=(i,j), dims=(0,1))
            edge_mask = torch.logical_or(edge_mask, shifted == -1)

    # 更新可见性掩码
    edge_faces = torch.unique(pix_to_face[edge_mask])
    edge_vertices = torch.unique(faces[edge_faces])
    vis_mask[edge_vertices] = 0.0

    return vis_mask


def image2vid(images, vid_path):

    w, h = images[0].size
    videodims = (w, h)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(vid_path, fourcc, len(images) / 5.0, videodims)
    for image in images:
        video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    video.release()


def query_color(verts, faces, image, device, predicted_color):
    """query colors from points and image

    Args:
        verts ([B, 3]): [query verts]
        faces ([M, 3]): [query faces]
        image ([B, 3, H, W]): [full image]

    Returns:
        [np.float]: [return colors]
    """

    verts = verts.float().to(device)
    faces = faces.long().to(device)
    predicted_color=predicted_color.to(device)
    (xy, z) = verts.split([2, 1], dim=1)
    visibility = get_visibility_color(xy, z, faces[:, [0, 2, 1]]).flatten()
    uv = xy.unsqueeze(0).unsqueeze(2)  # [B, N, 2]
    uv = uv * torch.tensor([1.0, -1.0]).type_as(uv)
    colors = (torch.nn.functional.grid_sample(
        image, uv, align_corners=True)[0, :, :, 0].permute(1, 0) +
              1.0) * 0.5 * 255.0
    colors[visibility == 0.0]=(predicted_color* 255.0)[visibility == 0.0]

    return colors.detach().cpu()


class cleanShader(torch.nn.Module):

    def __init__(self, device="cpu", cameras=None, blend_params=None):
        super().__init__()
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams(
        )

    def forward(self, fragments, meshes, **kwargs):
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of TexturedSoftPhongShader"

            raise ValueError(msg)

        # get renderer output
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = blending.softmax_rgb_blend(texels,
                                            fragments,
                                            blend_params,
                                            znear=-256,
                                            zfar=256)

        return images
    
    
class Render:

    def __init__(self, device, size=512,):
        self.device = device
        self.size = size

        # camera setting
        self.dis = 100.0
        self.scale = 100.0
        self.mesh_y_center = 0.0

        self.reload_cam()

        self.type = "color"

        self.mesh = None
        self.deform_mesh = None
        self.pcd = None
        self.renderer = None
        self.meshRas = None

        self.uv_rasterizer = Pytorch3dRasterizer(self.size)

    def reload_cam(self):

        self.cam_pos = [
            (0, self.mesh_y_center, self.dis),
            (self.dis, self.mesh_y_center, 0),
            (0, self.mesh_y_center, -self.dis),
            (-self.dis, self.mesh_y_center, 0),
            (0,self.mesh_y_center+self.dis,0),
            (0,self.mesh_y_center-self.dis,0),
        ]

    def get_camera(self, cam_id):
        
        if cam_id == 4:
            R, T = look_at_view_transform(
                eye=[self.cam_pos[cam_id]],
                at=((0, self.mesh_y_center, 0), ),
                up=((0, 0, 1), ),
            )
        elif cam_id == 5:
            R, T = look_at_view_transform(
                eye=[self.cam_pos[cam_id]],
                at=((0, self.mesh_y_center, 0), ),
                up=((0, 0, 1), ),
            )

        else:
            R, T = look_at_view_transform(
                eye=[self.cam_pos[cam_id]],
                at=((0, self.mesh_y_center, 0), ),
                up=((0, 1, 0), ),
            )

        camera = FoVOrthographicCameras(
            device=self.device,
            R=R,
            T=T,
            znear=100.0,
            zfar=-100.0,
            max_y=100.0,
            min_y=-100.0,
            max_x=100.0,
            min_x=-100.0,
            scale_xyz=(self.scale * np.ones(3), ),
        )

        return camera

    def init_renderer(self, camera, type="clean_mesh", bg="gray"):

        if "mesh" in type:

            # rasterizer
            self.raster_settings_mesh = RasterizationSettings(
                image_size=self.size,
                blur_radius=np.log(1.0 / 1e-4) * 1e-7,
                faces_per_pixel=30,
            )
            self.meshRas = MeshRasterizer(
                cameras=camera, raster_settings=self.raster_settings_mesh)

        if bg == "black":
            blendparam = BlendParams(1e-4, 1e-4, (0.0, 0.0, 0.0))
        elif bg == "white":
            blendparam = BlendParams(1e-4, 1e-8, (1.0, 1.0, 1.0))
        elif bg == "gray":
            blendparam = BlendParams(1e-4, 1e-8, (0.5, 0.5, 0.5))

        if type == "ori_mesh":

            lights = PointLights(
                device=self.device,
                ambient_color=((0.8, 0.8, 0.8), ),
                diffuse_color=((0.2, 0.2, 0.2), ),
                specular_color=((0.0, 0.0, 0.0), ),
                location=[[0.0, 200.0, 0.0]],
            )

            self.renderer = MeshRenderer(
                rasterizer=self.meshRas,
                shader=SoftPhongShader(
                    device=self.device,
                    cameras=camera,
                    lights=None,
                    blend_params=blendparam,
                ),
            )

        if type == "silhouette":
            self.raster_settings_silhouette = RasterizationSettings(
                image_size=self.size,
                blur_radius=np.log(1.0 / 1e-4 - 1.0) * 5e-5,
                faces_per_pixel=50,
                cull_backfaces=True,
            )

            self.silhouetteRas = MeshRasterizer(
                cameras=camera,
                raster_settings=self.raster_settings_silhouette)
            self.renderer = MeshRenderer(rasterizer=self.silhouetteRas,
                                         shader=SoftSilhouetteShader())

        if type == "pointcloud":
            self.raster_settings_pcd = PointsRasterizationSettings(
                image_size=self.size, radius=0.006, points_per_pixel=10)

            self.pcdRas = PointsRasterizer(
                cameras=camera, raster_settings=self.raster_settings_pcd)
            self.renderer = PointsRenderer(
                rasterizer=self.pcdRas,
                compositor=AlphaCompositor(background_color=(0, 0, 0)),
            )

        if type == "clean_mesh":

            self.renderer = MeshRenderer(
                rasterizer=self.meshRas,
                shader=cleanShader(device=self.device,
                                   cameras=camera,
                                   blend_params=blendparam),
            )

   

    def get_rgb_image(self,mesh, cam_ids=[0, 2], bg='gray'):

        images = []
        for cam_id in range(len(self.cam_pos)):
            if cam_id in cam_ids:
                self.init_renderer(self.get_camera(cam_id), "clean_mesh", bg)
                if len(cam_ids) == 4:
                    rendered_img = (self.renderer(meshes[0])[0:1, :, :, :3].permute(0, 3, 1, 2) -
                                    0.5) * 2.0
                else:
                    rendered_img = (self.renderer(meshes[0])[0:1, :, :, :3].permute(0, 3, 1, 2) -
                                    0.5) * 2.0
                if cam_id == 2 and len(cam_ids) == 2:
                    rendered_img = torch.flip(rendered_img, dims=[3])
                images.append(rendered_img)

        return images




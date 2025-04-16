from pytorch3d.structures import Meshes, Pointclouds
import torch.nn.functional as F
import torch
from kaolin.ops.mesh import check_sign, face_normals
from kaolin.metrics.trianglemesh import point_to_mesh_distance
import numpy as np
import json
import os.path as osp
import _pickle as cPickle

def face_vertices(vertices, faces):
    """ 
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) *
                     nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, vertices.shape[-1]))

    return vertices[faces.long()]

def barycentric_coordinates_of_projection(points, vertices):
    ''' https://github.com/MPI-IS/mesh/blob/master/mesh/geometry/barycentric_coordinates_of_projection.py
    '''
    """Given a point, gives projected coords of that point to a triangle
    in barycentric coordinates.
    See
        **Heidrich**, Computing the Barycentric Coordinates of a Projected Point, JGT 05
        at http://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf
    
    :param p: point to project. [B, 3]
    :param v0: first vertex of triangles. [B, 3]
    :returns: barycentric coordinates of ``p``'s projection in triangle defined by ``q``, ``u``, ``v``
            vectorized so ``p``, ``q``, ``u``, ``v`` can all be ``3xN``
    """
    #(p, q, u, v)
    v0, v1, v2 = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    p = points

    q = v0
    u = v1 - v0
    v = v2 - v0
    n = torch.cross(u, v)
    s = torch.sum(n * n, dim=1)
    # If the triangle edges are collinear, cross-product is zero,
    # which makes "s" 0, which gives us divide by zero. So we
    # make the arbitrary choice to set s to epsv (=numpy.spacing(1)),
    # the closest thing to zero
    s[s == 0] = 1e-6
    oneOver4ASquared = 1.0 / s
    w = p - q
    b2 = torch.sum(torch.cross(u, w) * n, dim=1) * oneOver4ASquared
    b1 = torch.sum(torch.cross(w, v) * n, dim=1) * oneOver4ASquared
    weights = torch.stack((1 - b1 - b2, b1, b2), dim=-1)
    # check barycenric weights
    # p_n = v0*weights[:,0:1] + v1*weights[:,1:2] + v2*weights[:,2:3]
    return weights

class PointFeat:

    def __init__(self, verts, faces):

        # verts [B, N_vert, 3]
        # faces [B, N_face, 3]
        # triangles [B, N_face, 3, 3]

        self.Bsize = verts.shape[0]
        self.mesh = Meshes(verts, faces)
        self.device = verts.device
        self.faces = faces

        # SMPL has watertight mesh, but SMPL-X has two eyeballs and open mouth
        # 1. remove eye_ball faces from SMPL-X: 9928-9383, 10474-9929
        # 2. fill mouth holes with 30 more faces

        if verts.shape[1] == 10475:
            faces = faces[:, ~SMPLX().smplx_eyeball_fid_mask]
            mouth_faces = (torch.as_tensor(
                SMPLX().smplx_mouth_fid).unsqueeze(0).repeat(
                    self.Bsize, 1, 1).to(self.device))
            self.faces = torch.cat([faces, mouth_faces], dim=1).long()

        self.verts = verts
        self.triangles = face_vertices(self.verts, self.faces)

    def get_face_normals(self):
        return face_normals(self.verts, self.faces)
    
    def get_nearest_point(self,points):
        # points [1, N, 3]
        # find nearest point on mesh

        #devices = points.device
        points=points.squeeze(0)
        nn_class=NN(X=self.verts.squeeze(0),Y=self.verts.squeeze(0),p=2)
        nearest_points,nearest_points_ind=nn_class.predict(points)
        
        # closest_triangles = torch.gather(
        #     self.triangles, 1,
        #     pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
        # bary_weights = barycentric_coordinates_of_projection(
        #     points.view(-1, 3), closest_triangles)
        
        # bary_weights=F.normalize(bary_weights, p=2, dim=1)

        # normals = face_normals(self.triangles)

        # # make the lenght of the normal is 1
        # normals = F.normalize(normals, p=2, dim=2)


        # # get the normal of the closest triangle
        # closest_normals = torch.gather(
        #     normals, 1,
        #     pts_ind[:, :, None].expand(-1, -1, 3)).view(-1, 3)
        

        return nearest_points,nearest_points_ind  # on cpu

    def query_barycentirc_feats(self,points,feats):
        # feats [B,N,C]

        residues, pts_ind, _ = point_to_mesh_distance(points, self.triangles)
        closest_triangles = torch.gather(
            self.triangles, 1,
            pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
        bary_weights = barycentric_coordinates_of_projection(
            points.view(-1, 3), closest_triangles)

        feat_arr=feats
        feat_dim = feat_arr.shape[-1]
        feat_tri = face_vertices(feat_arr, self.faces)        
        closest_feats = torch.gather(   # query点距离最近的face的三个点的feature
                    feat_tri, 1,
                    pts_ind[:, :, None,
                            None].expand(-1, -1, 3,
                                         feat_dim)).view(-1, 3, feat_dim)
        pts_feats = ((closest_feats *
                        bary_weights[:, :, None]).sum(1).unsqueeze(0)) # 用barycentric weight加权求和
        return pts_feats.view(self.Bsize,-1,feat_dim)
    
    def query_vis(self, points,vis, ):

        # points [B, N, 3]
        # feats {'feat_name': [B, N, C]}

        del_keys = ["smpl_verts", "smpl_faces", "smpl_joint","smpl_sample_id"]

        residues, pts_ind, _ = point_to_mesh_distance(points, self.triangles)
        closest_triangles = torch.gather(
            self.triangles, 1,
            pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
        bary_weights = barycentric_coordinates_of_projection(
            points.view(-1, 3), closest_triangles)

        out_dict = {}
        feat_dim = vis.shape[-1]
        feat_arr = vis
                
        feat_tri = face_vertices(feat_arr, self.faces)
        closest_feats = torch.gather(feat_tri, 1, pts_ind[:, :, None, None].expand(-1, -1, 3,feat_dim)).view(-1, 3, feat_dim)  # query点距离最近的face的三个点的feature
        pts_feats = ((closest_feats * bary_weights[:, :, None]).sum(1).unsqueeze(0)) # 用barycentric weight加权求和

        vis_out = pts_feats.ge(1e-1).float()
        vis_out = vis_out.view(self.Bsize, -1,  vis_out.shape[-1])
        return vis_out
    
    def query(self, points, feats={}):

        # points [B, N, 3]
        # feats {'feat_name': [B, N, C]}

        del_keys = ["smpl_verts", "smpl_faces", "smpl_joint","smpl_sample_id"]

        residues, pts_ind, _ = point_to_mesh_distance(points, self.triangles)
        closest_triangles = torch.gather(
            self.triangles, 1,
            pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
        bary_weights = barycentric_coordinates_of_projection(
            points.view(-1, 3), closest_triangles)

        out_dict = {}

        for feat_key in feats.keys():

            if feat_key in del_keys:
                continue

            elif feats[feat_key] is not None:
                feat_arr = feats[feat_key]
                feat_dim = feat_arr.shape[-1]
                feat_tri = face_vertices(feat_arr, self.faces)
                closest_feats = torch.gather(   # query点距离最近的face的三个点的feature
                    feat_tri, 1,
                    pts_ind[:, :, None,
                            None].expand(-1, -1, 3,
                                         feat_dim)).view(-1, 3, feat_dim)
                pts_feats = ((closest_feats *
                              bary_weights[:, :, None]).sum(1).unsqueeze(0)) # 用barycentric weight加权求和
                out_dict[feat_key.split("_")[1]] = pts_feats

            else:
                out_dict[feat_key.split("_")[1]] = None

        if "sdf" in out_dict.keys():
            pts_dist = torch.sqrt(residues) / torch.sqrt(torch.tensor(3))
            pts_signs = 2.0 * (
                check_sign(self.verts, self.faces[0], points).float() - 0.5)
            pts_sdf = (pts_dist * pts_signs).unsqueeze(-1)
            out_dict["sdf"] = pts_sdf

        if "vis" in out_dict.keys():
            out_dict["vis"] = out_dict["vis"].ge(1e-1).float()

        if "norm" in out_dict.keys():
            pts_norm = out_dict["norm"] * torch.tensor([-1.0, 1.0, -1.0]).to(
                self.device)
            out_dict["norm"] = F.normalize(pts_norm, dim=2)

        if "cmap" in out_dict.keys():
            out_dict["cmap"] = out_dict["cmap"].clamp_(min=0.0, max=1.0)

        for out_key in out_dict.keys():
            out_dict[out_key] = out_dict[out_key].view(
                self.Bsize, -1, out_dict[out_key].shape[-1])

        return out_dict
    
class SMPLX():

    def __init__(self):

        self.current_dir = "./data/smpl_related"  # new smplx file in ECON folder

        self.smpl_verts_path = osp.join(self.current_dir,
                                        "smpl_data/smpl_verts.npy")
        self.smpl_faces_path = osp.join(self.current_dir,
                                        "smpl_data/smpl_faces.npy")
        self.smplx_verts_path = osp.join(self.current_dir,
                                         "smpl_data/smplx_verts.npy")
        self.smplx_faces_path = osp.join(self.current_dir,
                                         "smpl_data/smplx_faces.npy")
        self.cmap_vert_path = osp.join(self.current_dir,
                                       "smpl_data/smplx_cmap.npy")

        self.smplx_to_smplx_path = osp.join(self.current_dir,
                                            "smpl_data/smplx_to_smpl.pkl")

        self.smplx_eyeball_fid = osp.join(self.current_dir,
                                          "smpl_data/eyeball_fid.npy")
        self.smplx_fill_mouth_fid = osp.join(self.current_dir,
                                             "smpl_data/fill_mouth_fid.npy")

        self.smplx_faces = np.load(self.smplx_faces_path)
        self.smplx_verts = np.load(self.smplx_verts_path)
        self.smpl_verts = np.load(self.smpl_verts_path)
        self.smpl_faces = np.load(self.smpl_faces_path)

        self.smplx_eyeball_fid_mask = np.load(self.smplx_eyeball_fid)
        self.smplx_mouth_fid = np.load(self.smplx_fill_mouth_fid)

        self.smplx_to_smpl = cPickle.load(open(self.smplx_to_smplx_path, 'rb'))

        self.model_dir = osp.join(self.current_dir, "models")
        self.tedra_dir = osp.join(self.current_dir, "../tedra_data")



        # copy from econ
        self.smplx_flame_vid_path = osp.join(
            self.current_dir, "smpl_data/FLAME_SMPLX_vertex_ids.npy"
        )
        self.smplx_mano_vid_path = osp.join(self.current_dir, "smpl_data/MANO_SMPLX_vertex_ids.pkl")
        self.smpl_vert_seg_path = osp.join(
            osp.dirname(__file__), "../../lib/common/smpl_vert_segmentation.json"
        )
        self.front_flame_path = osp.join(self.current_dir, "smpl_data/FLAME_face_mask_ids.npy")
        self.smplx_vertex_lmkid_path = osp.join(
            self.current_dir, "smpl_data/smplx_vertex_lmkid.npy"
        )

        self.smplx_vertex_lmkid = np.load(self.smplx_vertex_lmkid_path)
        self.smpl_vert_seg = json.load(open(self.smpl_vert_seg_path))
        self.smpl_mano_vid = np.concatenate(
            [
                self.smpl_vert_seg["rightHand"], self.smpl_vert_seg["rightHandIndex1"],
                self.smpl_vert_seg["leftHand"], self.smpl_vert_seg["leftHandIndex1"]
            ]
        )

        self.smplx_mano_vid_dict = np.load(self.smplx_mano_vid_path, allow_pickle=True)
        self.smplx_mano_vid = np.concatenate(
            [self.smplx_mano_vid_dict["left_hand"], self.smplx_mano_vid_dict["right_hand"]]
        )
        self.smplx_flame_vid = np.load(self.smplx_flame_vid_path, allow_pickle=True)
        self.smplx_front_flame_vid = self.smplx_flame_vid[np.load(self.front_flame_path)]


        # hands
        self.smplx_mano_vertex_mask = torch.zeros(self.smplx_verts.shape[0], ).index_fill_(
            0, torch.tensor(self.smplx_mano_vid), 1.0
        )
        self.smpl_mano_vertex_mask = torch.zeros(self.smpl_verts.shape[0], ).index_fill_(
            0, torch.tensor(self.smpl_mano_vid), 1.0
        )

         # face
        self.front_flame_vertex_mask = torch.zeros(self.smplx_verts.shape[0], ).index_fill_(
            0, torch.tensor(self.smplx_front_flame_vid), 1.0
        )
        self.eyeball_vertex_mask = torch.zeros(self.smplx_verts.shape[0], ).index_fill_(
            0, torch.tensor(self.smplx_faces[self.smplx_eyeball_fid_mask].flatten()), 1.0
        )


        self.ghum_smpl_pairs = torch.tensor(
            [
                (0, 24), (2, 26), (5, 25), (7, 28), (8, 27), (11, 16), (12, 17), (13, 18), (14, 19),
                (15, 20), (16, 21), (17, 39), (18, 44), (19, 36), (20, 41), (21, 35), (22, 40),
                (23, 1), (24, 2), (25, 4), (26, 5), (27, 7), (28, 8), (29, 31), (30, 34), (31, 29),
                (32, 32)
            ]
        ).long()

        # smpl-smplx correspondence
        self.smpl_joint_ids_24 = np.arange(22).tolist() + [68, 73]
        self.smpl_joint_ids_24_pixie = np.arange(22).tolist() + [61 + 68, 72 + 68]
        self.smpl_joint_ids_45 = np.arange(22).tolist() + [68, 73] + np.arange(55, 76).tolist()

        self.extra_joint_ids = np.array(
            [
                61, 72, 66, 69, 58, 68, 57, 56, 64, 59, 67, 75, 70, 65, 60, 61, 63, 62, 76, 71, 72,
                74, 73
            ]
        )

        self.extra_joint_ids += 68

        self.smpl_joint_ids_45_pixie = (np.arange(22).tolist() + self.extra_joint_ids.tolist())
        

def distance_matrix(x, y=None, p = 2): #pairwise distance of vectors
    
    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    
    dist = torch.norm(x - y, dim=-1) if torch.__version__ >= '1.7.0' else torch.pow(x - y, p).sum(2)**(1/p)
    
    return dist    
        
class NN():

    def __init__(self, X = None, Y = None, p = 2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")
        
        dist=[]
        chunk=10000
        for i in range(0,x.shape[0],chunk):
            dist.append(distance_matrix(x[i:i+chunk], self.train_pts, self.p))
            
        dist = torch.cat(dist, dim=0)
        labels = torch.argmin(dist, dim=1)
        return self.train_label[labels],labels
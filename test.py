from models.Transformer import ViTVQ, ViTEncoder, ViTDecoder
import torch
models = ViTVQ(image_size=512,channels=9)

encoder = ViTEncoder(image_size=512, patch_size=16, dim=256,depth=8,heads=8,mlp_dim=2048,channels=9)
decoder = ViTDecoder(image_size=512, patch_size=16, dim=256,depth=3,heads=8,mlp_dim=2048)
test_tensor = torch.randn((1,9,512,512))
smpl_normals={
            "image_B":torch.randn((1,3,512,512)),
            "image_R":torch.randn((1,3,512,512)),
            "image_L":torch.randn((1,3,512,512))
        }

outs2 = models(test_tensor,smpl_normals)

print(outs2[0].shape)
print(outs2[1].shape)
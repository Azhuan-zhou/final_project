from facenet_pytorch import MTCNN
from PIL import Image
import torch
import  numpy as np
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device) 

def detect_face(image):
    """
    检测人脸
    :param image: PIL Image
    :return: box, prob
    """
    box, prob = mtcnn.detect(image)  # box: (4,), prob: float
    return box, prob

def main(cfg):
    scene_lists = [os.path.join(cfg.root_dir, scene,'imgs') for scene in os.listdir(cfg.root_dir)]
    for scene in scene_lists:
        view_dict = {}
        view_lists = os.listdir(scene)
        for view in view_lists:
            img_path = os.path.join(scene, view,'rgb.png')
            img = Image.open(img_path).convert('RGB')
            boxes, probs = detect_face(img)  # box: (4,), prob: float
            if boxes is not None:
                max_idx = np.argmax(probs)
                box = boxes[max_idx] 
                prob = probs[max_idx] 
                view_dict[view] = {'bbox': box, 'prob': prob}
            else:
                view_dict[view] = {'bbox': None, 'prob': 0}
            
        sorted_views = sorted(view_dict.items(), key=lambda x: (x[1]['prob'] if x[1]['prob'] is not None else -1), reverse=True)
        top10_views = dict(sorted_views[:10])
        save_path = os.path.dirname(scene)
        np.save(os.path.join(save_path,'face_info.npy'), top10_views)
        
            
            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    args = parser.parse_args()
    main(args)
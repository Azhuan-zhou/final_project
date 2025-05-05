"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
import argparse
import torch
import numpy as np
import os
import trimesh
from PIL import Image
import pickle
import cv2
from tqdm import tqdm
from smplx import SMPLX
import json
import shutil
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

SMPLX_PATH = 'data/body_models/smplx'

body_model = SMPLX(model_path=SMPLX_PATH, gender='male', use_pca=True, num_pca_comps=12,
                        flat_hand_mean=False, use_face_contour=True).to(device)
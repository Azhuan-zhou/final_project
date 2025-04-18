"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import argparse
import h5py
from tqdm import tqdm
from dataset.dataset_utils import generate_data

def main(args):

    obj_list = [os.path.join(args.input_path, x) for x in sorted(os.listdir(args.input_path)) if os.path.isdir(os.path.join(args.input_path, x))]
    for local_path in tqdm(obj_list):
        generate_data(local_path, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create dataset of sample points, images, and SMPLX from mesh files.')
  
    parser.add_argument("-i", "--input-path", default='data/THuman/new_thuman2.0', type=str, help="Aligned THuman2.0 folder")
    parser.add_argument("-o", "--output-path", default='data/THuman/THuman_dataset', type=str, help="Output path")
    parser.add_argument("--size", default=1024, type=int, help="Image size")

    parser.add_argument("--nsamples", default=1000000, type=int, help="Number of 3D points to sample")
    # Reduce the number samples if you don't have enough disk space, 1000000 points generates around 100GB of data

    parser.add_argument("--nviews", default=36, type=int, help="Number of views to render")
    parser.add_argument("--camera-mode", default='orth', type=str, help="Camera mode: orth | persp")
    parser.add_argument("--camera-sampling", default='uniform', type=str, help="Camera sampling: uniform | random")
    main(parser.parse_args())
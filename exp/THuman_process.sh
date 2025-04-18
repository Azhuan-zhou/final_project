input_path=/root/autodl-tmp/final_project/data/THuman
output_path=/root/autodl-tmp/final_project/data/THuman/new_thuman2.0
output_path_thuman=/root/autodl-tmp/final_project/data/THuman/THuman_dataset

python dataset/align_thuman2_0.py --input-path $input_path --output-path $output_path

python dataset/build_thuman_dataset.py --input-path $output_path --output-path $output_path_thuman

python dataset/detect_face.py --root_dir $output_path_thuma
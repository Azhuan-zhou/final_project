input_path=data/THuman
output_path=data/THuman/new_thuman_2.1

python dataset/align_thuman.py --input-path $input_path --output-path $output_path

python dataset/build_thuman_dataset.py --input-path $output_path --output-path $output_path
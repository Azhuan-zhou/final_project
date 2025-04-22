input_path=/root/autodl-tmp/final_project/data/THuman
output_path=/root/autodl-tmp/final_project/data/THuman/new_thuman2.0
output_path_thuman=/root/autodl-tmp/final_project/data/THuman/THuman_dataset

python script/align_thuman2_0.py --input-path $input_path --output-path $output_path
echo "---------------finishing alignment---------------"
echo "---------------Building dataset---------------"
python script/build_thuman_dataset.py --input-path $output_path --output-path $output_path_thuman
echo "---------------Finished---------------"
echo "---------------Detecting face---------------"
python script/detect_face.py --root_dir $output_path_thuma
echo "---------------Finished---------------"
echo "---------------Generating multi-view images---------------"
python script/gen_mv.py --config configs/gen_mv.yaml
echo "---------------Dataset is ready---------------"
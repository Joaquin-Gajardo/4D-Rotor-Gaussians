echo "Processing all WAT scenes"
base_folder="/workspace/4D-Rotor-Gaussians"
dataset_folder="$base_folder/data/WAT"
export CUDA_VISIBLE_DEVICES=0

for input_folder in "$dataset_folder"/*
do
    scene=$(basename $input_folder)

    output_path="$base_folder/outputs/WAT"
    model_path="$output_path/$scene/splatfacto"
    if [ -d "$model_path" ]; then
        latest_exp_name=$(ls $model_path | sort -r | head -n 1)
        exp_path=$model_path/$latest_exp_name
        if [ -f "$exp_path/results.json" ]; then
            echo "Metrics for scene $scene are already processed on $exp_path"
            continue
        fi
    fi
        
    echo "Processing scene $scene"
    
    # Train
    ns-train splatfacto-big --data $input_folder --pipeline.model.path $input_folder --viewer.websocket-port 6021 --output-dir $output_path --vis wandb
    
    # Get directory
    latest_exp_name=$(ls $model_path | sort -r | head -n 1)
    exp_path=$model_path/$latest_exp_name

    # Render and calculate metrics
    ns-render dataset --load_config $exp_path/config.yml --output-path $exp_path --split test
    python scripts/metrics.py $exp_path/test

done
echo "Done processing all WAT scenes"
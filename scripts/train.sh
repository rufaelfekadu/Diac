

# !/bin/bash

train_model() {
    local model_name=$1
    local train_data=$2
    local test_data=${3:-""}
    local use_asr=${4:-"False"}
    local pretrained_path=${5:-""}
    local load_text_branch=${6:-"False"}
    local save_dir=${7:-"results/${model_name}"}
    
    echo "Training model: $model_name with data: $train_data"

    # Check if model is already trained
    if [ -f "$save_dir/training_done.txt" ]; then
        echo "Model $model_name already trained. Skipping..."
        return 0
    fi

    # Create save directory if it doesn't exist
    mkdir -p "$save_dir"
    
    # Build command
    cmd="python train_lightning.py --config configs/${model_name}.yml --opts "
    cmd+="DATA.TRAIN_PATH '$train_data' "
    
    # Add test path if provided
    if [ -n "$test_data" ]; then
        cmd+="DATA.TEST_PATH '$test_data' "
    fi
    
    cmd+="MODEL.USE_ASR $use_asr "
    
    # Add pretrained path if provided
    if [ -n "$pretrained_path" ]; then
        cmd+="MODEL.PRETRAINED_PATH '$pretrained_path' "
        cmd+="MODEL.LOAD_TEXT_BRANCH_ONLY $load_text_branch "
    fi
    
    cmd+="TRAIN.SAVE_DIR '$save_dir'"
    
    eval $cmd
    touch "$save_dir/training_done.txt"
}


for model in "lstm" "transformer"
do
    # Train text-only models
    train_model "$model" "data/tashkeela/train.txt" "data/tashkeela/val.txt" "False" "" "False" "results/${model}-text-only/tashkeela"
    
    train_model "$model" "data/clartts/train_no_special.txt" "" "False" "" "False" "results/${model}-text-only/clartts"
    
    train_model "$model" "data/clartts/train_no_special.txt" "" "False" "results/${model}-text-only/tashkeela/tensorboard/version_0/checkpoints/best_model.ckpt" "True" "results/${model}-text-only/tashkeela+clartts"
    
    # Train text + asr models
    train_model "$model" "data/clartts/train_no_special.txt" "data/clartts/test_no_special.txt" "True" "" "False" "results/${model}-text+asr/clartts"
    
    train_model "$model" "data/clartts/train_no_special.txt" "" "True" "results/${model}-text-only/tashkeela/tensorboard/version_0/checkpoints/best_model.ckpt" "True" "results/${model}-text+asr/tashkeela+clartts"
    
    echo "Completed training for $model architecture"
done


run_inference() {

    local model=$1
    local dataset=$2
    local use_asr=$3
    local model_type=$4

    # Determine paths based on parameters
    local test_path="data/clartts/"
    test_path+=$([ "$use_asr" = "True" ] && echo "test_no_special.txt" || echo "ClarTTS_test_Ref.txt")
    
    local result_dir="results/${model}-${model_type}/${dataset}"
    local model_path="${result_dir}/tensorboard/version_0/checkpoints/best_model.ckpt"
    local output_path="${result_dir}/predictions.txt"
    
    echo "Running inference with settings:"
    echo "  Model: $model"
    echo "  Dataset: $dataset"
    echo "  Use ASR: $use_asr"
    echo "  Model Type: $model_type"
    
    python inference.py --config configs/${model}.yml --opts \
        DATA.TEST_PATH "$test_path" \
        MODEL.USE_ASR $use_asr \
        INFERENCE.MODEL_PATH "$model_path" \
        INFERENCE.OUTPUT_PATH "$output_path" \
        INFERENCE.USE_ASR $use_asr
}

run_evaluation() {
    local model=$1
    local model_type=$2
    local dataset=$3
    local test_file=$4
    local log_file="${5:-results/eval.log}"
    
    local prediction_path="results/${model}-${model_type}/${dataset}/predictions.txt"
    local test_path="data/clartts/${test_file}"
    
    echo "Running evaluation with settings:"
    echo "  Model: $model"
    echo "  Model Type: $model_type"
    echo "  Dataset: $dataset"
    echo "  Test file: $test_path"
    echo "  Log file: $log_file"
    
    python eval.py -ofp "$prediction_path" -tfp "$test_path" --log_file "$log_file"
}
for model in "lstm"
do
    echo "Running inference for model: $model"

    # text only
    # python inference.py --config configs/${model}.yml --opts \
    #     DATA.TEST_PATH 'data/clartts/ClarTTS_test_Ref.txt' \
    #     MODEL.USE_ASR False \
    #     INFERENCE.MODEL_PATH 'results/'${model}'-text-only/tashkeela/tensorboard/version_0/checkpoints/best_model.ckpt' \
    #     INFERENCE.OUTPUT_PATH 'results/'${model}'-text-only/tashkeela/predictions.txt' \
    #     INFERENCE.USE_ASR False

    # python inference.py --config configs/${model}.yml --opts \
    #     DATA.TEST_PATH 'data/clartts/ClarTTS_test_Ref.txt' \
    #     MODEL.USE_ASR False \
    #     INFERENCE.MODEL_PATH 'results/'${model}'-text-only/clartts/tensorboard/version_0/checkpoints/best_model.ckpt' \
    #     INFERENCE.OUTPUT_PATH 'results/'${model}'-text-only/clartts/predictions.txt' \
    #     INFERENCE.USE_ASR False

    # python inference.py --config configs/${model}.yml --opts \
    #     DATA.TEST_PATH 'data/clartts/ClarTTS_test_Ref.txt' \
    #     MODEL.USE_ASR False \
    #     INFERENCE.MODEL_PATH 'results/'${model}'-text-only/tashkeela+clartts/tensorboard/version_0/checkpoints/best_model.ckpt' \
    #     INFERENCE.OUTPUT_PATH 'results/'${model}'-text-only/tashkeela+clartts/predictions.txt' \
    #     INFERENCE.USE_ASR False

    # text + asr
    python inference.py --config configs/${model}.yml --opts \
        DATA.TEST_PATH 'data/clartts/test_no_special.txt' \
        MODEL.USE_ASR True \
        INFERENCE.MODEL_PATH 'results/'${model}'-text+asr/clartts/tensorboard/version_0/checkpoints/best_model.ckpt' \
        INFERENCE.OUTPUT_PATH 'results/'${model}'-text+asr/clartts/predictions.txt' \
        INFERENCE.USE_ASR True

    # python inference.py --config configs/${model}.yml --opts \
    #     DATA.TEST_PATH 'data/clartts/ClarTTS_test_Ref.txt' \
    #     MODEL.USE_ASR True \
    #     MODEL.PRETRAINED_PATH 'results/'${model}'-text+asr/tashkeela+clartts/tensorboard/version_0/checkpoints/best_model.ckpt' \
    #     INFERENCE.OUTPUT_PATH 'results/'${model}'-text+asr/tashkeela+clartts/predictions.txt' \
    #     INFERENCE.USE_ASR True
done

# evaluate
log_file="results/eval.log"
for model in "lstm"
do
    # python eval.py -ofp results/${model}-text-only/tashkeela/predictions.txt -tfp data/clartts/ClarTTS_test_Ref.txt --log_file $log_file
    # python eval.py -ofp results/${model}-text-only/clartts/predictions.txt -tfp data/clartts/ClarTTS_test_Ref.txt --log_file $log_file
    # python eval.py -ofp results/${model}-text-only/tashkeela+clartts/predictions.txt -tfp data/clartts/ClarTTS_test_Ref.txt --log_file $log_file
    python eval.py -ofp results/${model}-text+asr/clartts/predictions.txt -tfp data/clartts/test_2.txt --log_file $log_file
    # python eval.py -ofp results/${model}-text+asr/tashkeela+clartts/predictions.txt -tfp data/clartts/ClarTTS_test_Ref.txt --log_file $log_file
done
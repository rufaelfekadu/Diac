
#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

# Logging helpers
log() { printf "%s %s\n" "[INFO]" "$*"; }
err() { printf "%s %s\n" "[ERROR]" "$*" >&2; }

decode() {
    local model=$1
    local dataset=$2
    local use_asr=${3:-"False"}
    local model_type=${4:-"text-only"}
    local override=${5:-""}

    local model_name="${model}-${model_type}-${dataset}"
    local result_dir="results/${model}-${model_type}/${dataset}"

    # Choose a sensible default test path depending on ASR usage
    local test_path
    test_path="data/clartts/test+asr.txt"

    # Skip if already decoded, unless override is specified
    if [ -f "${result_dir}/inference.done" ] && [ "${override}" != "true" ]; then
        log "Model ${model_name} already decoded (found ${result_dir}/inference.done). Skipping..."
        return 0
    fi

    local model_path="${result_dir}/tensorboard/version_0/checkpoints/best_model.ckpt"
    local output_path="${result_dir}/predictions.txt"

    # Basic validation
    if [ ! -f "configs/${model}.yml" ]; then
        err "Config file configs/${model}.yml not found for model ${model}"
        return 3
    fi
    if [ ! -f "${model_path}" ]; then
        err "Model checkpoint ${model_path} not found. Run training first or check the path."
        return 4
    fi

    if ! mkdir -p "${result_dir}"; then
        err "Failed to create result directory: ${result_dir}"
        return 5
    fi

    log "Running inference with settings:"
    log "  Model: ${model}"
    log "  Dataset: ${dataset}"
    log "  Use ASR: ${use_asr}"
    log "  Model Type: ${model_type}"
    log "  Model path: ${model_path}"
    log "  Output path: ${output_path}"

    # Build command as an array to avoid word-splitting and shell injection issues
    local -a cmd
    cmd=(python inference.py --config "configs/${model}.yml" --opts)
    cmd+=("DATA.TEST_PATH" "${test_path}")
    cmd+=("MODEL.USE_ASR" "${use_asr}")
    cmd+=("INFERENCE.MODEL_PATH" "${model_path}")
    cmd+=("INFERENCE.OUTPUT_PATH" "${output_path}")
    cmd+=("INFERENCE.USE_ASR" "${use_asr}")

    log "Running command: ${cmd[*]}"

    if "${cmd[@]}"; then
        log "Inference completed successfully for ${model_name}. Creating ${result_dir}/inference.done"
        touch "${result_dir}/inference.done"
        return 0
    else
        local rc=$?
        err "Inference failed for ${model_name} (exit code ${rc}). See above for details."
        return ${rc}
    fi
}

evaluate() {
    local model=$1
    local model_type=$2
    local dataset=$3
    local test_file=$4

    local prediction_path="results/${model}-${model_type}/${dataset}/predictions.txt"
    local test_path="data/clartts/${test_file}"

    log "Running evaluation with settings:"
    log "  Model: ${model}"
    log "  Model Type: ${model_type}"
    log "  Dataset: ${dataset}"
    log "  Test file: ${test_path}"
    log "  Log file: ${log_file}"

    if [ ! -f "${prediction_path}" ]; then
        err "Predictions file not found: ${prediction_path}. Run inference first."
        return 6
    fi
    if [ ! -f "${test_path}" ]; then
        err "Test file not found: ${test_path}."
        return 7
    fi

    local -a cmd
    cmd=(python eval.py -ofp "${prediction_path}" -tfp "${test_path}" --log_file "${log_file}")
    log "Running: ${cmd[*]}"
    if "${cmd[@]}"; then
        log "Evaluation completed for ${model}-${model_type}/${dataset}"
        return 0
    else
        local rc=$?
        err "Evaluation failed for ${model}-${model_type}/${dataset} (exit code ${rc})"
        return ${rc}
    fi

}

# If executed directly, provide a small CLI to run decode or evaluate for a single experiment.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    if [ $# -lt 1 ]; then
        echo "Usage: $0 <command> [args...]"
        echo "Commands: decode <model> <dataset> [use_asr] [model_type] [override]" 
        echo "          evaluate <model> <model_type> <dataset> <test_file> [log_file]"
        exit 1
    fi
    cmd="$1"
    shift
    case "$cmd" in
        decode)
            decode "$@"
            exit $?
            ;;
        evaluate)
            evaluate "$@"
            exit $?
            ;;
        *)
            echo "Unknown command: $cmd"
            exit 2
            ;;
    esac
fi
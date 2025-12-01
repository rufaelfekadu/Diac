
#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

# Logging helpers
log() { printf "%s %s\n" "[INFO]" "$*"; }
err() { printf "%s %s\n" "[ERROR]" "$*" >&2; }

decode() {
    
    local test_path=$1
    local use_asr=${2:-"False"}
    local override=${3:-""}
    local result_dir=$4

    # result_dir is expected to be in the format model-modeltype/dataset and cannot be empty
    if [ -z "${result_dir}" ]; then
        err "result_dir argument is required"
        return 1
    fi


    # Extract parts from the result path
    
    # Extract model and model_type from the directory name part before the slash
    local dir_name=$(dirname "${result_dir}")
    local base_name=$(basename "${dir_name}")
    local model=$(echo "${base_name}" | cut -d'-' -f1)
    local model_type=$(echo "${base_name}" | cut -d'-' -f2-)
    local dataset=$(basename "${result_dir}")

    local model_name="${model}-${model_type}-${dataset}"


    # Choose a sensible default test path depending on ASR usage
    if [ -z "${test_path}" ]; then
        test_path="data/clartts/test.txt"
    fi

    # Skip if already decoded, unless override is specified
    # if [ -f "${result_dir}/inference.done" ] && [ "${override}" != "true" ]; then
    #     log "Model ${model_name} already decoded (found ${result_dir}/inference.done). Skipping..."
    #     return 0
    # fi

    # merge the dirs in the test path to one name using - as separator
    local pred_file=$(echo "${test_path}" | tr '/' '-' | sed 's/^-//' | sed 's/-$//' | sed 's/\.[^.]*$//')

    local latest_version=$(ls -d "${result_dir}/tensorboard/version_"* 2>/dev/null | sort -V | tail -n 1)
    local model_path="${latest_version}/checkpoints/best_model.ckpt"
    local output_path="${result_dir}/outs/${pred_file}"

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
    cmd+=("INFERENCE.OUTPUT_PATH" "${output_path}/pred.txt")
    cmd+=("INFERENCE.USE_ASR" "${use_asr}")

    log "Running command: ${cmd[*]}"

    if "${cmd[@]}"; then
        log "Inference completed successfully for ${model_name}. Creating ${result_dir}/inference.done"
        # touch "${result_dir}/inference.done"
        python src/diac/utils/prep_ref.py --input_file "${test_path}" -o "${output_path}"
        return 0
    else
        local rc=$?
        err "Inference failed for ${model_name} (exit code ${rc}). See above for details."
        return ${rc}
    fi
    
}

evaluate() {

    
    local test_path=${1:-""}
    local result_dir=$2

    # results-complete/transformer-text-only/tashkeela+arvoice
        local dir_name=$(dirname "${result_dir}")
        local base_name=$(basename "${dir_name}")
        local model=$(echo "${base_name}" | cut -d'-' -f1)
        local model_type=$(echo "${base_name}" | cut -d'-' -f2-)
        local dataset=$(basename "${result_dir}")
        local model_name="${model}-${model_type}-${dataset}"

    
    local pred_file=$(echo "${test_path}" | tr '/' '-' | sed 's/^-//' | sed 's/-$//' | sed 's/\.[^.]*$//')
    local prediction_path="${result_dir}/outs/${pred_file}/pred.txt"
    local reference_path="${result_dir}/outs/${pred_file}/ref.txt"

    # set default test file if not provided
    # if [ -z "${test_path}" ]; then
    #     test_path="${reference_path}"
    # fi

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
    cmd=(python src/diac/utils/eval.py -ofp "${prediction_path}" -tfp "${reference_path}" --log_file "${log_file}")
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
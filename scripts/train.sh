


#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

# Simple logging helpers used by the training functions
log() { printf "%s %s\n" "[INFO]" "$*"; }
err() { printf "%s %s\n" "[ERROR]" "$*" >&2; }

# train_model: function that runs the training command for a given model
train_model() {
    local model_name=$1
    local train_data=$2
    local val_data=${3:-""}
    local test_data=${4:-""}
    local use_asr=${5:-"False"}
    local pretrained_path=${6:-""}
    local load_text_branch=${7:-"False"}
    local override=${8:-"False"}
    local save_dir=${9:-"results/${model_name}"}

    log "Training model: ${model_name} with data: ${train_data}"

    # Check if model is already trained
    if [ -f "${save_dir}/training.done" ] && [ "${override}" = "False" ]; then
        log "Model ${model_name} already trained (found ${save_dir}/training.done). Skipping..."
        return 0
    fi

    # Validate config
    local config_file="configs/${model_name}.yml"
    if [ ! -f "${config_file}" ]; then
        err "Config file ${config_file} not found for model ${model_name}"
        return 3
    fi

    # Create save directory if it doesn't exist
    if ! mkdir -p "${save_dir}"; then
        err "Failed to create save directory: ${save_dir}"
        return 4
    fi

    # Build command safely as an array to avoid word-splitting and shell injection
    local -a cmd
    cmd=("python" "train_lightning.py" "--config" "${config_file}" "--opts")

    # Append options as separate arguments (Hydra-style overrides: KEY VALUE)
    cmd+=("DATA.TRAIN_PATH" "${train_data}")

    if [ -n "${val_data}" ]; then
        cmd+=("DATA.VAL_PATH" "${val_data}")
    fi
    if [ -n "${test_data}" ]; then
        cmd+=("DATA.TEST_PATH" "${test_data}")
    fi

    cmd+=("MODEL.USE_ASR" "${use_asr}")

    if [ -n "${pretrained_path}" ]; then
        cmd+=("MODEL.PRETRAINED_PATH" "${pretrained_path}")
        cmd+=("MODEL.LOAD_TEXT_BRANCH_ONLY" "${load_text_branch}")
    fi

    cmd+=("TRAIN.SAVE_DIR" "${save_dir}")

    # Print the assembled command for debugging
    log "Running command: ${cmd[*]}"

    # Execute the command and capture exit code
    if "${cmd[@]}"; then
        log "Training completed successfully for ${model_name}. Creating ${save_dir}/training.done"
        touch "${save_dir}/training.done"
        return 0
    else
        local rc=$?
        err "Training failed for ${model_name} (exit code ${rc}). See above for details."
        return ${rc}
    fi
}

# If this file is executed directly, allow running a single training job from CLI.
# Usage: ./train.sh <model_name> <train_data> [val_data] [test_data] [use_asr] [pretrained] [load_text_branch] [save_dir]
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    # Minimal CLI parsing (positional)
    if [ $# -lt 2 ]; then
        echo "Usage: $0 <model_name> <train_data> [val_data] [test_data] [use_asr] [pretrained] [load_text_branch] [save_dir]"
        exit 1
    fi
    train_model "$@"
    exit $?
fi


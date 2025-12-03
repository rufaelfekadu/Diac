
#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

# Sweep / orchestration entrypoint. This script contains the experiment sweep logic and
# uses multiprocessing helpers from `run.sh`. It also sources `train.sh` and `eval.sh`
# to access train_model, decode, and evaluate functions.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/run.sh"  # provides run_job, wait_for_all_jobs, logging helpers
source "${SCRIPT_DIR}/train.sh"
source "${SCRIPT_DIR}/eval.sh"

DATA_PATH="data"

avail_models=("transformer")
avail_model_types=("text-only" "text+asr") # text-only and text+asr
avail_pretrain_datasets=("tashkeela") 
avail_finetune_datasets=("clartts")
test_paths=("data/clartts/test.txt")
# CLI option parsing for sweep orchestration
STOP_ON_FAILURE=0
# Stage controls which stages to run:
#   0 = pretrain
#   1 = finetune
#   2 = decode
#   3 = evaluate
# By default run all stages (0..3)
STAGE=0
STOP_STAGE=3
OVERRIDE=""
OUTPUT_DIR="outputs/final-models"

while [[ ${#} -gt 0 ]]; do
    case "$1" in
        --stop-on-failure)
            STOP_ON_FAILURE=1
            shift
            ;;
        --override)
            OVERRIDE="true"
            shift
            ;;
        --stage)
            if [ -n "${2:-}" ]; then
                STAGE="$2"
                shift 2
            else
                err "--stage requires a numeric value"
                exit 1
            fi
            ;;
        --stop-stage|--stop_stage)
            if [ -n "${2:-}" ]; then
                STOP_STAGE="$2"
                shift 2
            else
                err "--stop-stage requires a numeric value"
                exit 1
            fi
            ;;
        --max-jobs)
            if [ -n "${2:-}" ]; then
                MAX_JOBS="$2"
                shift 2
            else
                err "--max-jobs requires a value"
                exit 1
            fi
            ;;
        --output-dir|--output_dir)
            if [ -n "${2:-}" ]; then
                OUTPUT_DIR="$2"
                shift 2
            else
                err "--output-dir requires a directory path"
                exit 1
            fi
            ;;
        --)
            shift
            break
            ;;
        -* )
            err "Unknown option: $1"
            exit 1
            ;;
        * )
            break
            ;;
    esac
done

    # validate stage variables are integers and sensible
    re='^[0-9]+$'
    if ! [[ "$STAGE" =~ $re ]]; then
        err "--stage must be a non-negative integer"
        exit 1
    fi
    if ! [[ "$STOP_STAGE" =~ $re ]]; then
        err "--stop-stage must be a non-negative integer"
        exit 1
    fi
    if [ "$STAGE" -gt "$STOP_STAGE" ]; then
        err "--stage ($STAGE) cannot be greater than --stop-stage ($STOP_STAGE)"
        exit 1
    fi

# Export STOP_ON_FAILURE for the multiprocessing helper to consult
export STOP_ON_FAILURE

# Concurrency settings: number of parallel jobs (already defaulted in run.sh but allow override)
MAX_JOBS=${MAX_JOBS:-${MAX_JOBS}}

log "$STAGE to $STOP_STAGE sweep starting with max ${MAX_JOBS} parallel jobs"

#################################
# Stage 0: pretrain
#################################
if [ "$STAGE" -le 0 ] && [ "$STOP_STAGE" -ge 0 ]; then
    # for each model, dataset, and model type, run training
    log "Runnig Pretraining"

    #  pretrain jobs
    for model in "${avail_models[@]}"; do
        for dataset in "${avail_pretrain_datasets[@]}"; do

            train_data="${DATA_PATH}/${dataset}/train.txt"
            test_data="${DATA_PATH}/${dataset}/test.txt"
            # check if val set exists; use it if so, otherwise pass empty string
            if [ -f "${DATA_PATH}/${dataset}/val.txt" ]; then
                val_data="${DATA_PATH}/${dataset}/val.txt"
            else
                val_data=""
            fi
            run_job "${model} - ${dataset} text-only" train_model "${model}" "${train_data}" "${val_data}" "${test_data}" "False" "" "" "" "${OUTPUT_DIR}/${model}-text-only/${dataset}" 
        done
    done
    
    log "Waiting for all pretraining jobs to complete"
    wait_for_all_jobs

else
    log "Skipping training stage (stage=${STAGE}, stop_stage=${STOP_STAGE})"
fi

###############################
# Stage 1: finetune
###############################
if [ "$STAGE" -le 1 ] && [ "$STOP_STAGE" -ge 1 ]; then
    log "Runnig Finetuning"
    for model in "${avail_models[@]}"; do
        for dataset in "${avail_finetune_datasets[@]}"; do
            for model_type in "${avail_model_types[@]}"; do
                train_data="${DATA_PATH}/${dataset}/train.txt"
                test_data="${DATA_PATH}/${dataset}/test.txt"
                pretrained_path="${OUTPUT_DIR}/${model}-text-only/tashkeela/tensorboard/version_0/checkpoints/best_model.ckpt"
                # check if val set exists; use it if so, otherwise pass empty string
                if [ -f "${DATA_PATH}/${dataset}/val.txt" ]; then
                    val_data="${DATA_PATH}/${dataset}/val.txt"
                else
                    val_data=""
                fi

                # set use_asr based on model_type
                use_asr="False"
                if [ "${model_type}" = "text+asr" ]; then
                    use_asr="True"
                fi

                # train from scratch on finetune dataset
                run_job "${model} - ${dataset} ${model_type}" \
                    train_model "${model}" "${train_data}" "${val_data}" "${test_data}" "${use_asr}" "" "True" "${OVERRIDE:-"False"}" "${OUTPUT_DIR}/${model}-${model_type}/${dataset}"

                # finetune from pretraining checkpoint
                run_job "${model} - tashkeela+${dataset} ${model_type}" \
                    train_model "${model}" "${train_data}" "${val_data}" "${test_data}" "${use_asr}" "${pretrained_path}" "True" "${OVERRIDE:-"False"}" "${OUTPUT_DIR}/${model}-${model_type}/tashkeela+${dataset}"
            done
        done
    done

    log "Waiting for all finetuning jobs to complete"
    wait_for_all_jobs
else
    log "Skipping finetune stage (stage=${STAGE}, stop_stage=${STOP_STAGE})"
fi

################################
# Stage 2: decode
################################
if [ "$STAGE" -le 2 ] && [ "$STOP_STAGE" -ge 2 ]; then
    log "Queueing inference (decode) jobs for available models/datasets"
    
    use_asr="False"
    for model_type in "${avail_model_types[@]}"; do
        for model in "${avail_models[@]}"; do
            for test_path in "${test_paths[@]}"; do
                for dataset in "${avail_finetune_datasets[@]}"; do
                    # update use_asr based on model_type
                    if [ "${model_type}" = "text+asr" ]; then
                        use_asr="True"
                    else
                        use_asr="False"
                    fi
                        run_job "${model} - ${dataset} ${model_type} decode" decode "${test_path}" "${use_asr}" "${OVERRIDE:-""}" "${OUTPUT_DIR}/${model}-${model_type}/${dataset}"
                        run_job "${model} - tashkeela+${dataset} ${model_type} decode" decode "${test_path}" "${use_asr}" "${OVERRIDE:-""}" "${OUTPUT_DIR}/${model}-${model_type}/tashkeela+${dataset}"
                    done
                for dataset in "${avail_pretrain_datasets[@]}"; do
                    if [ "${model_type}" = "text+asr" ]; then
                        continue
                    fi
                    run_job "${model} - ${dataset} ${model_type} decode" decode "${test_path}" "False" "${OVERRIDE:-""}" "${OUTPUT_DIR}/${model}-${model_type}/${dataset}"
                done
            done
        done
    done


    log "Waiting for inference jobs to complete"
    wait_for_all_jobs
else
    log "Skipping decode stage (stage=${STAGE}, stop_stage=${STOP_STAGE})"
fi

################################
# Stage 3: evaluation
################################
if [ "$STAGE" -le 3 ] && [ "$STOP_STAGE" -ge 3 ]; then
    log_file="${OUTPUT_DIR}/eval.log"
    mkdir -p "${OUTPUT_DIR}"
    rm -f "${log_file}" || true
    log "Queueing evaluation jobs (predictions must exist, decode step above will produce them)"

    for model_type in "${avail_model_types[@]}"; do
        # choose test file based on model_type

        for test_path in "${test_paths[@]}"; do
            for model in "${avail_models[@]}"; do
                # finetune datasets and tashkeela+dataset variants
                for dataset in "${avail_finetune_datasets[@]}"; do
                    run_job "${model} - eval ${dataset} ${model_type}" evaluate "${test_path}" "${OUTPUT_DIR}/${model}-${model_type}/${dataset}"
                    run_job "${model} - eval tashkeela+${dataset} ${model_type}" evaluate "${test_path}" "${OUTPUT_DIR}/${model}-${model_type}/tashkeela+${dataset}"
                done

                # pretrain datasets only for text-only
                if [ "${model_type}" != "text+asr" ]; then
                    for dataset in "${avail_pretrain_datasets[@]}"; do
                        run_job "${model} - eval ${dataset} ${model_type}" evaluate "${test_path}" "${OUTPUT_DIR}/${model}-${model_type}/${dataset}"
                    done
                fi
            done
        done
    done

    log "Waiting for evaluation jobs to complete"
    wait_for_all_jobs
else
    log "Skipping evaluation stage (stage=${STAGE}, stop_stage=${STOP_STAGE})"
fi

# Final summary: report job results and exit non-zero if any failed
FAILED=0
for pid in "${!JOB_RC[@]}"; do
    rc=${JOB_RC[$pid]}
    desc=${JOB_DESC[$pid]:-"(unknown)"}
    if [ "$rc" -ne 0 ]; then
        err "Job ${pid} (${desc}) failed with exit code ${rc}"
        FAILED=$((FAILED + 1))
    else
        log "Job ${pid} (${desc}) succeeded"
    fi
done

if [ "$FAILED" -ne 0 ]; then
    err "${FAILED} job(s) failed. See logs above."
    exit 10
fi
log "All sweep jobs completed successfully"

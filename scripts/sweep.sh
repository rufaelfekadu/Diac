
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

# CLI option parsing for sweep orchestration
STOP_ON_FAILURE=0
# stage controls which stages to run: 0=train, 1=decode, 2=eval
# By default run all stages
STAGE=0
STOP_STAGE=2
OVERRIDE=""
OUTPUT_DIR="results"
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

# Stage 0: training (base + finetune)
if [ "$STAGE" -le 0 ] && [ "$STOP_STAGE" -ge 0 ]; then
    for model in "lstm" "transformer"; do
        run_job "${model} - tashkeela text-only" train_model "${model}" "data/tashkeela/train.txt" "data/tashkeela/val.txt" "" "False" "" "False" "${OUTPUT_DIR}/${model}-text-only/tashkeela"
        run_job "${model} - clartts text-only" train_model "${model}" "data/clartts/train+asr.txt" "" "data/clartts/test+asr.txt" "False" "" "False" "${OUTPUT_DIR}/${model}-text-only/clartts"
        run_job "${model} - clartts text+asr" train_model "${model}" "data/clartts/train+asr.txt" "" "data/clartts/test+asr.txt" "True" "" "False" "${OUTPUT_DIR}/${model}-text+asr/clartts"
    done

    log "Waiting for all background training jobs to complete (max parallel jobs = ${MAX_JOBS})"
    wait_for_all_jobs

    # finetune jobs
    for model in "lstm" "transformer"; do
        run_job "${model} - tashkeela+clartts text-only (finetune)" train_model "${model}" "data/clartts/train+asr.txt" "" "data/clartts/test+asr.txt" "False" "${OUTPUT_DIR}/${model}-text-only/tashkeela/tensorboard/version_0/checkpoints/best_model.ckpt" "True" "${OUTPUT_DIR}/${model}-text-only/tashkeela+clartts"
        run_job "${model} - tashkeela+clartts text+asr (finetune)" train_model "${model}" "data/clartts/train+asr.txt" "" "data/clartts/test+asr.txt" "True" "${OUTPUT_DIR}/${model}-text-only/tashkeela/tensorboard/version_0/checkpoints/best_model.ckpt" "True" "${OUTPUT_DIR}/${model}-text+asr/tashkeela+clartts"
        log "Queued finetune training jobs for ${model} architecture"
    done

    wait_for_all_jobs
else
    log "Skipping training stage (stage=${STAGE}, stop_stage=${STOP_STAGE})"
fi

# Stage 1: decode
if [ "$STAGE" -le 1 ] && [ "$STOP_STAGE" -ge 1 ]; then
    log "Queueing inference (decode) jobs for available models/datasets"
    for model in "lstm" "transformer"; do
        run_job "${model} - tashkeela text-only decode" decode "${model}" "tashkeela" "False" "text-only" "${OVERRIDE:-""}" "${OUTPUT_DIR}/${model}-text-only/tashkeela"
        run_job "${model} - clartts text-only decode" decode "${model}" "clartts" "False" "text-only" "${OVERRIDE:-""}" "${OUTPUT_DIR}/${model}-text-only/clartts"
        run_job "${model} - tashkeela+clartts text-only decode" decode "${model}" "tashkeela+clartts" "False" "text-only" "${OVERRIDE:-""}" "${OUTPUT_DIR}/${model}-text-only/tashkeela+clartts"

        run_job "${model} - clartts text+asr decode" decode "${model}" "clartts" "True" "text+asr" "${OVERRIDE:-""}" "${OUTPUT_DIR}/${model}-text+asr/clartts"
        run_job "${model} - tashkeela+clartts text+asr decode" decode "${model}" "tashkeela+clartts" "True" "text+asr" "${OVERRIDE:-""}" "${OUTPUT_DIR}/${model}-text+asr/tashkeela+clartts"
    done

    log "Waiting for inference jobs to complete"
    wait_for_all_jobs
else
    log "Skipping decode stage (stage=${STAGE}, stop_stage=${STOP_STAGE})"
fi

# Stage 2: evaluation
if [ "$STAGE" -le 2 ] && [ "$STOP_STAGE" -ge 2 ]; then
    log_file="${OUTPUT_DIR}/eval.log"
    mkdir -p "${OUTPUT_DIR}"
    rm -f "${log_file}" || true
    log "Queueing evaluation jobs (predictions must exist, decode step above will produce them)"
    for model in "lstm" "transformer"; do
        run_job "${model} - eval tashkeela text-only" evaluate "${model}" "text-only" "tashkeela" "test.txt" "${OUTPUT_DIR}/${model}-text-only/tashkeela"
        run_job "${model} - eval clartts text-only" evaluate "${model}" "text-only" "clartts" "test.txt" "${OUTPUT_DIR}/${model}-text-only/clartts"
        run_job "${model} - eval tashkeela+clartts text-only" evaluate "${model}" "text-only" "tashkeela+clartts" "test.txt" "${OUTPUT_DIR}/${model}-text-only/tashkeela+clartts"

        run_job "${model} - eval clartts text+asr" evaluate "${model}" "text+asr" "clartts" "test.txt" "${OUTPUT_DIR}/${model}-text+asr/clartts"
        run_job "${model} - eval tashkeela+clartts text+asr" evaluate "${model}" "text+asr" "tashkeela+clartts" "test.txt" "${OUTPUT_DIR}/${model}-text+asr/tashkeela+clartts"
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
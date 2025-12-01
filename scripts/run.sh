#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

# Multiprocessing helpers used by orchestration scripts (sweep.sh or other callers).
# This file should only be sourced; it does not enqueue or run experiment-specific jobs itself.

# Logging helpers (exported so callers can use them)
log() { printf "%s %s\n" "[INFO]" "$*"; }
err() { printf "%s %s\n" "[ERROR]" "$*" >&2; }

# Basic error handler for ERR trap (caller can override by setting their own trap)
error_handler() {
    local exit_code=${1:-1}
    local line_no=${2:-unknown}
    err "Script failed with exit code ${exit_code} at line ${line_no}"
    exit ${exit_code}
}
trap 'error_handler $? $LINENO' ERR

# Default concurrency; callers may override before sourcing or set MAX_JOBS in env
MAX_JOBS=${MAX_JOBS:-$(nproc 2>/dev/null || echo 2)}

# Temporary directory to store per-job exit codes
STATUS_DIR=$(mktemp -d -t diac-train-status.XXXX)
PIDS=()
declare -A JOB_DESC=()
declare -A JOB_RC=()
# declare -A PIDS=()

# Cleanup handler to kill background jobs and remove status dir
cleanup() {
    log "Cleaning up background jobs and temporary files"
    # kill any remaining background jobs
    if [ ${#PIDS[@]} -gt 0 ]; then
        log "Killing ${#PIDS[@]} background job(s)"
        for p in "${PIDS[@]}"; do
            if kill -0 "$p" 2>/dev/null; then
                kill "$p" || true
            fi
        done
    fi
    rm -rf "${STATUS_DIR}" || true
}
trap cleanup EXIT

# Start a job in the background and throttle to MAX_JOBS
# Usage: run_job "description" <command...> <save_dir>
run_job() {
    local desc=$1
    shift

    # The last argument is expected to be the save_dir for logging
    local save_dir
    save_dir="${@: -1}"
    if [ -z "${save_dir}" ]; then
        err "Unable to determine save_dir for job '${desc}' (last arg empty)"
        return 2
    fi

    # Prepare log file inside the save_dir
    if ! mkdir -p "${save_dir}/logs"; then
        err "Failed to create logs directory in ${save_dir}"
        return 3
    fi

    # Determine job type based on command name (first arg)
    local first_cmd="$1"
    local job_type="unknown"
    case "${first_cmd}" in
        train_model) job_type="train" ;;
        decode) job_type="decode" ;;
        evaluate) job_type="eval" ;;
    esac

    local logfile="${save_dir}/logs/${job_type}-$(date +%s)-${RANDOM}.log"


    #  check for the training.done file to avoid overwriting existing jobs
    if [ "${job_type}" = "train" ] && [ -f "${save_dir}/training.done" ]; then
        # log "Skipping job '${desc}' as training.done already exists in ${save_dir}"
        return 0
    fi
    
    log "Queueing job: ${desc}"

    # Run the provided command in a background subshell that writes its exit code to a status file
    (
        set -o pipefail
        # Redirect stdout/stderr only to logfile (do not mirror to main console)
        exec >>"${logfile}" 2>&1
        # Execute the requested command (function or external)
        "$@"
        rc=$?
        # Write exit code to per-job status file named after this subshell's PID
        echo "$rc" > "${STATUS_DIR}/${BASHPID}.status"
    ) &

    local pid=$!
    PIDS+=("$pid")
    JOB_DESC["$pid"]="${desc}"

    # Throttle: if we've reached MAX_JOBS, wait for at least one to finish
    while [ "${#PIDS[@]}" -ge "${MAX_JOBS}" ]; do
        reap_one_job || sleep 1
    done
}

# Reap a single completed job if available. Returns 0 if a job was reaped.
reap_one_job() {
    for i in "${!PIDS[@]}"; do
        pid=${PIDS[i]}
        status_file="${STATUS_DIR}/${pid}.status"
        if [ -f "$status_file" ]; then
            rc=$(cat "$status_file" || echo 1)
            JOB_RC["$pid"]=$rc
            desc=${JOB_DESC[$pid]:-"(unknown)"}
            if [ "$rc" -eq 0 ]; then
                log "Job ${pid} (${desc}) finished successfully"
            else
                err "Job ${pid} (${desc}) failed with exit code ${rc}"
            fi
            # remove pid from PIDS array
            unset 'PIDS[i]'
            rm -f "$status_file"

            # If configured to stop on first failure, exit immediately
            if [ "$rc" -ne 0 ] && [ "${STOP_ON_FAILURE:-0}" -ne 0 ]; then
                err "Stopping early due to failure and --stop-on-failure set"
                cleanup
                exit "$rc"
            fi

            return 0
        fi
    done
    return 1
}

# Wait for all outstanding jobs to complete
wait_for_all_jobs() {
    while [ ${#PIDS[@]} -gt 0 ]; do
        if ! reap_one_job; then
            # no job ready yet, wait briefly
            sleep 1
        fi
    done
}

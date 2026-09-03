#!/bin/bash
# Eval script for: batch8_0.2_retry
# Checkpoints (in test order): best_action_l2 -> step_100k -> step_80k -> step_60k -> step_40k -> step_20k
# 6 checkpoints total, each tested sequentially

env
__conda_setup="$('/root/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/root/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/root/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

sudo apt update
sudo apt install xvfb -y
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools

conda activate qwen3

echo "--------------Sh Start------------------"

USE_JUICEFS=0

export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000

export S3_ENDPOINT="https://obs.cn-southwest-2.myhuaweicloud.com"
export S3_USE_HTTPS="0"
export ACCESS_KEY_ID="HPUACJ8EWHNONWBP1PGQ"
export SECRET_ACCESS_KEY="rx3ITEWElWS8dZnPHmFMmMiiGkQ2pk5FguQ0d1QN"


echo "----------------- copy RoboTwin code -----------------"
RoboTwinCodePath="obs://yw-2030-gy/external/wwx1513998/code/robotwin_threeclip"
rm -rf /cache/wwx1513998/RoboTwin
mkdir -p /cache/wwx1513998/RoboTwin
python -c "import moxing as mox; mox.file.copy_parallel('$RoboTwinCodePath', '/cache/wwx1513998/RoboTwin/')"
cd /cache/wwx1513998/RoboTwin
echo "--------------- copy RoboTwin code done ---------------"


echo "------------------- set envs -------------------"
envs_name=RoboTwin
if [ ! -d "/root/miniconda3/envs/$envs_name" ]; then
    python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-2030-gy/external/wwx1484778/envs/$envs_name.tar.gz','/cache/envs/$envs_name.tar.gz')"
    mkdir -p /root/miniconda3/envs/$envs_name
    tar -xf /cache/envs/$envs_name.tar.gz -C /root/miniconda3/envs
fi
conda activate $envs_name

echo "/cache/wwx1513998/RoboTwin/envs/curobo/src" > /root/miniconda3/envs/RoboTwin/lib/python3.10/site-packages/__editable__.nvidia_curobo-0.0.0.pth
python script/update_embodiment_config_path.py
echo "----------------- set envs done -----------------"


echo "------------------ Render Test ------------------"
bash rendernew.sh
python script/test_render.py
echo "---------------- Render Test Done ----------------"


echo "----------------- Download pretrained weights -----------------"
conda activate qwen3
rm -rf /cache/wx1513998/pretrained_models
mkdir -p /cache/wx1513998/pretrained_models/Wan2.2-TI2V-5B
mkdir -p /cache/wx1513998/pretrained_models/Qwen3-VL-2B-Instruct

WanPath="obs://yw-2030-gy/external/wwx1484778/Wan2.2-TI2V-5B"
VLMPath="obs://yw-2030-gy/external/wwx1484778/Qwen3-VL-2B-Instruct"
python -c "import moxing as mox; mox.file.copy_parallel('$WanPath', '/cache/wx1513998/pretrained_models/Wan2.2-TI2V-5B')"
python -c "import moxing as mox; mox.file.copy_parallel('$VLMPath', '/cache/wx1513998/pretrained_models/Qwen3-VL-2B-Instruct')"

conda activate $envs_name
echo "--------------- Download pretrained weights done ---------------"


# ============================================================================
# Define all checkpoints to test (ordered: best first, then steps descending)
# ============================================================================
CKPT_OBS_BASE="obs://yw-2030-gy/external/wwx1513998/checkpoint/motus_robotwin_subtask_Motus_threeclip_20w_batch8_0.2_retry"
RUN_TAG="batch8_0.2_retry"

CKPT_LIST=(
    "best_action_l2"
    "checkpoint_step_100000"
    "checkpoint_step_80000"
    "checkpoint_step_60000"
    "checkpoint_step_40000"
    "checkpoint_step_20000"
)

echo "========================================================="
echo "Testing ${#CKPT_LIST[@]} checkpoints for $RUN_TAG"
echo "========================================================="


# ============================================================================
# Sequential evaluation: download ckpt -> evaluate -> upload logs -> cleanup
# ============================================================================
for CKPT_NAME in "${CKPT_LIST[@]}"; do
    echo ""
    echo "========================================================="
    echo "Evaluating: $RUN_TAG / $CKPT_NAME"
    echo "========================================================="

    # Download checkpoint
    CKPT_OBS="${CKPT_OBS_BASE}/${CKPT_NAME}/pytorch_model"
    CKPT_LOCAL="/cache/wx1513998/eval_checkpoints/${RUN_TAG}/${CKPT_NAME}/pytorch_model"

    echo "Downloading checkpoint from $CKPT_OBS ..."
    conda activate qwen3
    rm -rf "$CKPT_LOCAL"
    mkdir -p "$CKPT_LOCAL"
    python -c "import moxing as mox; mox.file.copy_parallel('$CKPT_OBS', '$CKPT_LOCAL')"

    if [ ! -d "$CKPT_LOCAL" ]; then
        echo "ERROR: Failed to download checkpoint for $CKPT_NAME, skipping..."
        continue
    fi
    echo "Checkpoint downloaded to $CKPT_LOCAL"

    # Write paths_config.yml
    conda activate $envs_name
    CONFIG_DIR="/cache/wwx1513998/RoboTwin/policy/MotusWanVlmDirectMask"
    cat > "$CONFIG_DIR/paths_config.yml" << EOF
robotwin_root: "/cache/wwx1513998/RoboTwin"
conda_env: "RoboTwin"
checkpoint_path: "$CKPT_LOCAL"
wan_path: "/cache/wx1513998/pretrained_models/Wan2.2-TI2V-5B"
vlm_path: "/cache/wx1513998/pretrained_models/Qwen3-VL-2B-Instruct"
gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
task_config: "demo_randomized"
seed: 42
tasks_file: "tasks_all_new.txt"
EOF

    echo "paths_config.yml written for $CKPT_NAME"

    # Make sure Xvfb is running
    if ! pgrep -x Xvfb > /dev/null; then
        Xvfb :99 -screen 0 1024x768x24 &>/dev/null &
        export DISPLAY=:99
        echo "Xvfb restarted"
    fi

    # Run evaluation
    cd /cache/wwx1513998/RoboTwin
    conda activate $envs_name
    bash policy/MotusWanVlmDirectMask/auto_eval_new.sh
    EVAL_EXIT_CODE=$?

    echo "Evaluation of $CKPT_NAME exit code: $EVAL_EXIT_CODE"

    # Upload logs
    echo "------- Uploading eval logs for $CKPT_NAME -------"
    LOG_DIR=$(ls -dt /cache/wwx1484778/RoboTwin/MotusWanVlmDirectMask/logs_wan_vlm_auto_* 2>/dev/null | head -1)

    if [ -n "$LOG_DIR" ] && [ -d "$LOG_DIR" ]; then
        LOG_BASENAME=$(basename "$LOG_DIR")
        EVAL_OBS_PATH="obs://yw-2030-gy/external/wwx1513998/eval_logs/${RUN_TAG}/${CKPT_NAME}/$LOG_BASENAME"
        echo "Uploading $LOG_DIR -> $EVAL_OBS_PATH"

        conda activate qwen3
        python -c "import moxing as mox; mox.file.copy_parallel('$LOG_DIR', '$EVAL_OBS_PATH')"
        if [ $? -eq 0 ]; then
            echo "Upload logs done: $EVAL_OBS_PATH"
        else
            echo "Moxing upload failed, trying OBS SDK..."
            /root/miniconda3/envs/motus/bin/python -c "
from obs import ObsClient; import os
obsClient = ObsClient(access_key_id='HPUACJ8EWHNONWBP1PGQ', secret_access_key='rx3ITEWElWS8dZnPHmFMmMiiGkQ2pk5FguQ0d1QN', server='obs.cn-southwest-2.myhuaweicloud.com')
bucket='yw-2030-gy'; local_dir='$LOG_DIR'; obs_prefix='external/wwx1513998/eval_logs/${RUN_TAG}/${CKPT_NAME}/$LOG_BASENAME'
count=0
for root, dirs, files in os.walk(local_dir):
    for f in files:
        lp = os.path.join(root, f)
        rp = os.path.relpath(lp, local_dir)
        obsClient.uploadFile(bucket, obs_prefix+'/'+rp, lp)
        count+=1
obsClient.close()
print(f'Upload done (obs SDK): {count} files -> obs://{bucket}/{obs_prefix}')
"
        fi

        # Also upload eval_result if it exists
        EVAL_RESULT_DIR="/cache/wwx1484778/RoboTwin/eval_result"
        if [ -d "$EVAL_RESULT_DIR" ]; then
            RESULT_OBS_PATH="obs://yw-2030-gy/external/wwx1513998/eval_logs/${RUN_TAG}/${CKPT_NAME}/eval_result"
            echo "Uploading eval results -> $RESULT_OBS_PATH"
            python -c "import moxing as mox; mox.file.copy_parallel('$EVAL_RESULT_DIR', '$RESULT_OBS_PATH')" 2>/dev/null
        fi
    else
        echo "WARNING: No log directory found for $CKPT_NAME"
    fi

    # Free disk: remove local checkpoint copy after evaluation
    echo "Cleaning up local checkpoint copy for $CKPT_NAME..."
    rm -rf "/cache/wx1513998/eval_checkpoints/${RUN_TAG}/${CKPT_NAME}"

    echo "------- $CKPT_NAME evaluation and upload done -------"
done

# Cleanup Xvfb
pkill Xvfb 2>/dev/null || echo "Xvfb already stopped"

echo ""
echo "========================================================="
echo "All ${#CKPT_LIST[@]} checkpoints for $RUN_TAG completed!"
echo "========================================================="

exit 0

#!/bin/bash
# Launch script for WebSocket-based MotusWanVlmDirectMask RoboTwin evaluation
#
# This starts both the server (GPU inference) and client (RoboTwin env) processes.
# The server loads the model once, and the client connects to it for each inference step.
#
# Usage:
#   bash run_ws_eval.sh [GPU_ID] [TASK_NAME]
#   bash run_ws_eval.sh              # defaults: GPU 0, all tasks
#   bash run_ws_eval.sh 0 scan_object  # GPU 0, single task

set -e

GPU_ID=${1:-0}
TASK_NAME=${2:-""}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOTUS_ROOT="$(dirname "$SCRIPT_DIR")"

# ============== Configuration (edit these paths) ==============
MODEL_CONFIG="${MOTUS_ROOT}/RobboTwin_test/policy/MotusWanVlmDirectMask/utils/robotwin_wan_vlm.yml"
CKPT_DIR="${MOTUS_ROOT}/checkpoints/ADS/checkpoints/robotwin_wan_vlm_mask_stage2/motus_wan_vlm_robotwin_bs8_lr5e-05/checkpoint_step_80000/pytorch_model"
WAN_PATH="/home/ma-user/work/l30083605/Models/Wan2.2-TI2V-5B"
VLM_PATH="/home/ma-user/work/l30083605/Models/Qwen3-VL-2B-Instruct"
HOST="localhost"
PORT=6790
SEED=42
TASK_CONFIG="demo_randomized"
TEST_NUM=20

# ============== Validate paths ==============
if [ ! -f "$MODEL_CONFIG" ]; then
    echo "Error: Model config not found: $MODEL_CONFIG"
    exit 1
fi

if [ ! -d "$CKPT_DIR" ]; then
    echo "Error: Checkpoint not found: $CKPT_DIR"
    exit 1
fi

if [ ! -d "$WAN_PATH" ]; then
    echo "Error: WAN path not found: $WAN_PATH"
    exit 1
fi

if [ ! -d "$VLM_PATH" ]; then
    echo "Error: VLM path not found: $VLM_PATH"
    exit 1
fi

# ============== Build client args ==============
CLIENT_ARGS="--host ${HOST} --port ${PORT} --seed ${SEED} --task_config ${TASK_CONFIG} --test_num ${TEST_NUM}"

if [ -n "$TASK_NAME" ]; then
    CLIENT_ARGS="${CLIENT_ARGS} --task_name ${TASK_NAME}"
fi

# ============== Start server ==============
echo "Starting WebSocket inference server on GPU ${GPU_ID}..."
echo "  Model config: ${MODEL_CONFIG}"
echo "  Checkpoint:   ${CKPT_DIR}"
echo "  WAN path:     ${WAN_PATH}"
echo "  VLM path:     ${VLM_PATH}"
echo "  Address:      ws://${HOST}:${PORT}"
echo ""

CUDA_VISIBLE_DEVICES=${GPU_ID} python "${SCRIPT_DIR}/dobot_websocket_vlm_server.py" \
    --model_config "${MODEL_CONFIG}" \
    --ckpt_dir "${CKPT_DIR}" \
    --wan_path "${WAN_PATH}" \
    --vlm_path "${VLM_PATH}" \
    --device "cuda:0" \
    --host "0.0.0.0" \
    --port ${PORT} \
    &

SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in $(seq 1 60); do
    if curl -s "http://${HOST}:${PORT}/healthz" > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "Error: Server process died"
        exit 1
    fi
    echo "  Waiting... (${i}/60)"
    sleep 2
done

# ============== Start client ==============
echo ""
echo "Starting WebSocket client..."
echo "  Client args: ${CLIENT_ARGS}"
echo ""

python "${SCRIPT_DIR}/dobot_websocket_vlm_client.py" ${CLIENT_ARGS}

CLIENT_EXIT=$?

# ============== Cleanup ==============
echo ""
echo "Shutting down server (PID: ${SERVER_PID})..."
kill ${SERVER_PID} 2>/dev/null || true

echo "Client exited with code: ${CLIENT_EXIT}"
exit ${CLIENT_EXIT}

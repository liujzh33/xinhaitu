#!/usr/bin/env python3
"""
WebSocket client for MotusWanVlmDirectMask RoboTwin evaluation.

This client runs the RoboTwin environment, sends observations to the
WebSocket inference server, receives predicted actions, and executes them.

Usage (terminal 2, after server is running):
    python dobot_websocket_vlm_client.py \
      --host localhost \
      --port 6790 \
      --task_name scan_object \
      --task_config demo_randomized \
      --seed 42

Or run all tasks:
    python dobot_websocket_vlm_client.py \
      --host localhost \
      --port 6790 \
      --tasks_file /path/to/tasks_all_new.txt \
      --task_config demo_randomized \
      --seed 42
"""

import argparse
import importlib
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import msgpack
import numpy as np
import torch
import websockets.sync.client
from PIL import Image

# RoboTwin paths (used later, imports are deferred until cwd is set)
ROBBOTWIN_ROOT = str(Path(__file__).resolve().parents[1] / "RobboTwin_test")

log = logging.getLogger("dobot_websocket_vlm_client")


# ============== MessagePack helpers (same as server) ==============

def _pack_default(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "data": obj.tobytes(),
        }
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Unsupported type for MessagePack: {type(obj)!r}")


def pack_message(data: Any) -> bytes:
    return msgpack.packb(data, default=_pack_default, use_bin_type=True)


def _object_hook(obj: Dict[Any, Any]) -> Any:
    if obj.get("__ndarray__"):
        array = np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"]))
        return array.reshape(obj["shape"]).copy()
    return obj


def unpack_message(data: bytes) -> Any:
    return msgpack.unpackb(data, raw=False, object_hook=_object_hook)


# ============== WebSocket Client ==============

class InferenceClient:
    """WebSocket client that communicates with the MotusWanVlm inference server."""

    def __init__(self, host: str = "localhost", port: int = 6790, reconnect_interval_s: float = 5.0):
        if host.startswith("ws://") or host.startswith("wss://"):
            self.uri = host
        else:
            self.uri = f"ws://{host}"
        if port is not None and ":" not in self.uri.removeprefix("ws://").removeprefix("wss://"):
            self.uri += f":{port}"

        self.reconnect_interval_s = reconnect_interval_s
        self._ws = None
        self._metadata = None

    def connect(self) -> Dict[str, Any]:
        """Connect to server and return metadata."""
        log.info("Waiting for WebSocket server at %s ...", self.uri)
        while True:
            try:
                self._ws = websockets.sync.client.connect(self.uri, compression=None, max_size=None)
                metadata_raw = self._ws.recv()
                if isinstance(metadata_raw, str):
                    raise RuntimeError(f"Expected binary metadata, got text:\n{metadata_raw}")
                self._metadata = unpack_message(metadata_raw)
                log.info("Connected to server metadata: %s", self._metadata)
                return self._metadata
            except ConnectionRefusedError:
                log.info("Server is not ready, retrying in %.1fs", self.reconnect_interval_s)
                time.sleep(self.reconnect_interval_s)

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Send observation and receive inference result."""
        if self._ws is None:
            raise RuntimeError("Not connected. Call connect() first.")
        self._ws.send(pack_message(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error from inference server:\n{response}")
        return unpack_message(response)

    def close(self) -> None:
        if self._ws is not None:
            self._ws.close()
            self._ws = None


# ============== Observation encoding ==============

def encode_observation(observation: Dict[str, Any], video_height: int = 384, video_width: int = 320) -> Dict[str, Any]:
    """
    Encode a RoboTwin observation into the format expected by the server.

    RoboTwin observation format:
        observation.observation.head_camera.rgb   -> top camera (np.ndarray HxWx3)
        observation.observation.left_camera.rgb   -> left wrist camera
        observation.observation.right_camera.rgb  -> right wrist camera
        observation.joint_action.vector           -> 14-dim robot state (qpos)

    Server expects:
        images: dict with "top", "left_wrist", "right_wrist" keys (np.ndarray uint8)
        state: 14-dim float32 vector
        instruction: task instruction string
    """
    obs_data = observation.get("observation", observation)

    # Extract camera images
    images = {}
    if "head_camera" in obs_data:
        images["top"] = obs_data["head_camera"]["rgb"]
    if "left_camera" in obs_data:
        images["left_wrist"] = obs_data["left_camera"]["rgb"]
    if "right_camera" in obs_data:
        images["right_wrist"] = obs_data["right_camera"]["rgb"]

    # Fallback for other observation formats
    if not images:
        if "head_camera" in observation:
            images["top"] = observation["head_camera"]
        if "image" in observation:
            images["top"] = observation["image"]

    # Ensure uint8
    for key in images:
        img = images[key]
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                if np.issubdtype(img.dtype, np.floating):
                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            images[key] = img

    # Extract robot state
    state = observation.get("joint_action", {}).get("vector", None)
    if state is None and "state" in observation:
        state = observation["state"]
    if state is not None:
        state = np.asarray(state, dtype=np.float32).reshape(-1)

    result = {"images": images}
    if state is not None:
        result["state"] = state
    return result


# ============== RoboTwin Evaluation Loop ==============

def load_task_env(task_name: str):
    """Load RoboTwin task environment. Must be called after setup_robottwin_env()."""
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def setup_robottwin_env():
    """Setup RoboTwin environment paths and working directory. Must be called before env imports."""
    os.chdir(ROBBOTWIN_ROOT)
    os.environ["PYTHONPATH"] = ROBBOTWIN_ROOT + ":" + os.environ.get("PYTHONPATH", "")
    if ROBBOTWIN_ROOT not in sys.path:
        sys.path.insert(0, ROBBOTWIN_ROOT)
    policy_dir = os.path.join(ROBBOTWIN_ROOT, "policy")
    desc_utils = os.path.join(ROBBOTWIN_ROOT, "description", "utils")
    if policy_dir not in sys.path:
        sys.path.append(policy_dir)
    if desc_utils not in sys.path:
        sys.path.append(desc_utils)


def eval_single_task(
    client: InferenceClient,
    task_name: str,
    task_config: str,
    seed: int,
    test_num: int = 20,
    instruction_type: str = "unseen",
    save_dir: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Evaluate a single task using the WebSocket inference server.

    Returns (success_count, total_episodes).
    """
    from envs import CONFIGS_PATH
    import yaml

    # Load task config
    with open(os.path.join(ROBBOTWIN_ROOT, "task_config", f"{task_config}.yml"), "r") as f:
        args = yaml.safe_load(f)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["eval_mode"] = True
    args["policy_name"] = "MotusWanVlmDirectMask"
    args["ckpt_setting"] = "websocket_server"

    # Load embodiment config
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r") as f:
        _embodiment_types = yaml.safe_load(f)

    embodiment_type = args.get("embodiment")
    if len(embodiment_type) == 1:
        robot_file = _embodiment_types[embodiment_type[0]]["file_path"]
        args["left_robot_file"] = robot_file
        args["right_robot_file"] = robot_file
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = _embodiment_types[embodiment_type[0]]["file_path"]
        args["right_robot_file"] = _embodiment_types[embodiment_type[1]]["file_path"]
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False

    def get_embodiment_config(robot_file):
        config_file = os.path.join(robot_file, "config.yml")
        with open(config_file, "r") as f:
            return yaml.safe_load(f)

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    # Camera config
    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r") as f:
        _camera_config = yaml.safe_load(f)
    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    # Create task environment
    log.info("Loading task environment: %s ...", task_name)
    TASK_ENV = load_task_env(task_name)
    log.info("Task environment loaded: %s", task_name)

    # Setup save dir
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if save_dir is None:
        save_dir = os.path.join(ROBBOTWIN_ROOT, "eval_result", task_name,
                                "MotusWanVlmDirectMask_WS", task_config, current_time)
    os.makedirs(save_dir, exist_ok=True)

    # Setup video save dir (matching eval_policy.py behavior)
    if args.get("eval_video_log") and not args.get("eval_video_save_dir"):
        args["eval_video_save_dir"] = save_dir

    # Import instruction generator
    from generate_episode_instructions import generate_episode_descriptions

    # Get server metadata to know dimensions
    metadata = client._metadata or {}
    common = metadata.get("common", {})
    state_dim = common.get("state_dim", 14)
    video_height = common.get("video_height", 384)
    video_width = common.get("video_width", 320)

    # Evaluation loop
    st_seed = 100000 * (1 + seed)
    suc_count = 0
    total_count = 0
    episode_id = 0
    now_seed = st_seed

    while suc_count < test_num:
        render_freq = args.get("render_freq", 0)
        args["render_freq"] = 0

        # Try to set up demo
        try:
            TASK_ENV.setup_demo(now_ep_num=episode_id, seed=now_seed, is_test=True, **args)
            episode_info = TASK_ENV.play_once()
            TASK_ENV.close_env()
        except Exception as e:
            log.warning("Setup demo failed for seed %d: %s", now_seed, str(e)[:200])
            TASK_ENV.close_env()
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        # Check if expert can solve
        if not (TASK_ENV.plan_success and TASK_ENV.check_success()):
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        # Valid episode found
        suc_seed_idx = suc_count

        # Setup environment
        TASK_ENV.setup_demo(now_ep_num=episode_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(task_name, episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)

        # Video recording (matching eval_policy.py: uses TASK_ENV.eval_video_path)
        ffmpeg_proc = None
        if TASK_ENV.eval_video_path is not None:
            video_size = f"{args['head_camera_w']}x{args['head_camera_h']}"
            video_path = os.path.join(TASK_ENV.eval_video_path, f"episode{total_count}.mp4")
            ffmpeg_proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                 "-video_size", video_size, "-framerate", "10", "-i", "-",
                 "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23", video_path],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg_proc)

        # Run policy
        succ = False
        step_count = 0
        try:
            while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
                observation = TASK_ENV.get_obs()

                # Encode observation for server
                encoded_obs = encode_observation(observation, video_height, video_width)
                encoded_obs["instruction"] = instruction

                # Send to server and get actions
                result = client.infer(encoded_obs)

                # Extract predicted actions
                actions = np.asarray(result.get("predicted_actions", []), dtype=np.float32)
                if actions.ndim == 1:
                    actions = actions.reshape(1, -1)

                # Execute actions one by one
                for action in actions:
                    TASK_ENV.take_action(action, action_type="qpos")

                step_count += 1

                if TASK_ENV.eval_success:
                    succ = True
                    break

        except Exception as e:
            log.error("Episode %d failed with error: %s", episode_id, str(e)[:200])
            traceback.print_exc()

        # Cleanup video (matching eval_policy.py)
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            suc_count += 1
            log.info("\033[92mSuccess!\033[0m")
        else:
            log.info("\033[91mFail!\033[0m")

        total_count += 1
        episode_id += 1
        now_seed += 1
        TASK_ENV.close_env(clear_cache=(total_count % args.get("clear_cache_freq", 10) == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        log.info(
            f"\033[93m{task_name}\033[0m | "
            f"Success: \033[96m{suc_count}/{total_count}\033[0m "
            f"=> \033[95m{round(suc_count/total_count*100, 1)}%\033[0m, "
            f"seed: \033[90m{now_seed}\033[0m"
        )

    return suc_count, total_count


# ============== Main ==============

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WebSocket RoboTwin client for MotusWanVlmDirectMask")

    # Server connection
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=6790, help="Server port")

    # Task configuration
    parser.add_argument("--task_name", type=str, default=None, help="Single task name to evaluate")
    parser.add_argument("--tasks_file", type=str, default=None, help="File with task names (one per line)")
    parser.add_argument("--task_config", type=str, default="demo_randomized", help="Task config name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test_num", type=int, default=20, help="Number of successful episodes per task")
    parser.add_argument("--instruction_type", type=str, default="unseen", help="Instruction type")

    # Output
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory for results")

    # Connection options
    parser.add_argument("--reconnect_interval", type=float, default=5.0, help="Reconnect interval in seconds")
    parser.add_argument("--gpu", type=str, default=None, help="GPU id for client SAPIEN rendering (default: disable GPU)")

    return parser


def main() -> int:
    # Force unbuffered output so logs appear immediately
    import functools
    sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), 'w', buffering=1)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", force=True)
    args = build_argparser().parse_args()

    # curobo motion planner needs CUDA to create tensors, so the client must
    # have GPU access.  Use --gpu to pick a specific device; default keeps
    # whatever CUDA_VISIBLE_DEVICES was set by the caller.
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        log.info("Client GPU set to CUDA_VISIBLE_DEVICES=%s", args.gpu)
    else:
        log.info("Client using default CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)"))

    # Determine tasks to run
    tasks = []
    if args.task_name:
        tasks = [args.task_name]
    elif args.tasks_file:
        with open(args.tasks_file, "r") as f:
            tasks = [line.strip() for line in f if line.strip()]
    else:
        # Default tasks from tasks_all_new.txt
        default_tasks_file = os.path.join(
            ROBBOTWIN_ROOT, "policy", "MotusWanVlmDirectMask", "tasks_all_new.txt"
        )
        if os.path.exists(default_tasks_file):
            with open(default_tasks_file, "r") as f:
                tasks = [line.strip() for line in f if line.strip()]
        else:
            log.error("No task specified. Use --task_name or --tasks_file")
            return 1

    if not tasks:
        log.error("No tasks to evaluate")
        return 1

    # Connect to server
    client = InferenceClient(host=args.host, port=args.port, reconnect_interval_s=args.reconnect_interval)
    metadata = client.connect()
    log.info("Server metadata: %s", metadata)

    # Setup RoboTwin environment (must be done before importing env modules)
    # Set DISPLAY before any SAPIEN imports
    os.environ["DISPLAY"] = ":99"

    setup_robottwin_env()

    # Start Xvfb for SAPIEN offscreen rendering (skip if already running)
    if not os.system("pgrep -x Xvfb > /dev/null 2>&1"):
        log.info("Xvfb already running, skipping start")
    else:
        os.system("Xvfb :99 -screen 0 1024x768x24 &>/dev/null &")
        time.sleep(1)
        log.info("Xvfb started on :99")

    # Run evaluation for each task
    all_results = {}
    for task_name in tasks:
        log.info(f"\n{'='*60}")
        log.info(f"Starting task: {task_name}")
        log.info(f"{'='*60}")

        try:
            suc, total = eval_single_task(
                client=client,
                task_name=task_name,
                task_config=args.task_config,
                seed=args.seed,
                test_num=args.test_num,
                instruction_type=args.instruction_type,
                save_dir=args.save_dir,
            )
            rate = round(suc / total * 100, 1) if total > 0 else 0.0
            all_results[task_name] = {"success": suc, "total": total, "rate": rate}
            log.info(f"Task {task_name}: {suc}/{total} = {rate}%")
        except Exception as e:
            log.error(f"Task {task_name} failed: {e}", exc_info=True)
            all_results[task_name] = {"success": 0, "total": 0, "rate": 0.0, "error": str(e)}

    # Print summary
    log.info(f"\n{'='*60}")
    log.info("EVALUATION SUMMARY")
    log.info(f"{'='*60}")
    total_success = 0
    total_episodes = 0
    for task_name, result in all_results.items():
        rate = result.get("rate", 0.0)
        log.info(f"  {task_name}: {result.get('success', 0)}/{result.get('total', 0)} = {rate}%")
        total_success += result.get("success", 0)
        total_episodes += result.get("total", 0)

    if total_episodes > 0:
        overall_rate = round(total_success / total_episodes * 100, 1)
        log.info(f"  Overall: {total_success}/{total_episodes} = {overall_rate}%")

    # Save results
    if args.save_dir:
        summary_path = os.path.join(args.save_dir, "evaluation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        log.info(f"Results saved to {summary_path}")

    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

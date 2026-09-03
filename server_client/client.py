#!/usr/bin/env python3
"""
Motus Inference API Client

Test client for `inference/real_world/Motus/server.py`.
"""

import argparse
import base64
import json
import os
import sys
import tempfile
import time
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import torch
from PIL import Image


PROJ_ROOT = Path(__file__).resolve().parents[3]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))


class MotusAPIClient:
    """Client for the Motus inference API."""

    def __init__(self, base_url: str = "http://localhost:6789") -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    def health_check(self) -> Dict[str, Any]:
        response = self.session.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()

    def get_model_info(self) -> Dict[str, Any]:
        response = self.session.get(f"{self.base_url}/model_info", timeout=10)
        response.raise_for_status()
        return response.json()

    def encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def create_random_image_base64(self, width: int = 384, height: int = 320) -> str:
        random_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        image = Image.fromarray(random_array)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def run_inference(
        self,
        instruction: str,
        images: Optional[List[str]] = None,
        image: Optional[str] = None,
        image_path: Optional[str] = None,
        state: Optional[List[float]] = None,
        proprio_data: Optional[List[List[float]]] = None,
        t5_embeddings: Optional[Any] = None,
        t5_embeddings_path: Optional[str] = None,
        t5_embeddings_dir: Optional[str] = None,
        auto_find_t5_embeddings: bool = True,
        num_inference_steps: Optional[int] = None,
        instruction_prefix: Optional[str] = None,
        return_frame_grid: bool = False,
        save_output_path: Optional[str] = None,
        timeout: int = 300,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "instruction": instruction,
            "auto_find_t5_embeddings": auto_find_t5_embeddings,
            "return_frame_grid": return_frame_grid,
        }

        if images is not None:
            payload["images"] = images
        if image is not None:
            payload["image"] = image
        if image_path is not None:
            payload["image_path"] = image_path
        if state is not None:
            payload["state"] = state
        if proprio_data is not None:
            payload["proprio_data"] = proprio_data
        if t5_embeddings is not None:
            payload["t5_embeddings"] = t5_embeddings
        if t5_embeddings_path is not None:
            payload["t5_embeddings_path"] = t5_embeddings_path
        if t5_embeddings_dir is not None:
            payload["t5_embeddings_dir"] = t5_embeddings_dir
        if num_inference_steps is not None:
            payload["num_inference_steps"] = num_inference_steps
        if instruction_prefix is not None:
            payload["instruction_prefix"] = instruction_prefix
        if save_output_path is not None:
            payload["save_output_path"] = save_output_path

        response = self.session.post(f"{self.base_url}/inference", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()


def _read_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _read_json_float_list(path: str) -> List[float]:
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError("state json 必须是一维数组，例如 [0.1, 0.2, ...]")
    return [float(x) for x in data]


def _parse_csv_to_float_list(csv_str: str) -> List[float]:
    values = [x.strip() for x in csv_str.split(",") if x.strip()]
    return [float(x) for x in values]


def _save_frame_grid_if_present(result: Dict[str, Any], output_path: Optional[str]) -> None:
    frame_grid_b64 = result.get("frame_grid_image")
    if not frame_grid_b64 or not output_path:
        return
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(base64.b64decode(frame_grid_b64))


def _tensor_image_to_base64(image_tensor: torch.Tensor) -> str:
    if image_tensor.ndim != 3:
        raise ValueError(f"Expected image tensor [C,H,W], got {tuple(image_tensor.shape)}")
    image_np = image_tensor.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()
    image_uint8 = (image_np * 255.0).astype(np.uint8)
    image = Image.fromarray(image_uint8)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _save_temp_t5_embedding(embedding: torch.Tensor) -> str:
    tmp = tempfile.NamedTemporaryFile(prefix="motus_t5_", suffix=".pt", delete=False)
    tmp.close()
    torch.save(embedding.detach().cpu(), tmp.name)
    return tmp.name


def _load_lerobot_dataset(
    dataset_root: str,
    dataset_config: str,
    repo_id: Optional[str],
    embodiment_type: str,
    t5_wan_path: Optional[str],
    enable_t5_fallback: bool,
    max_episodes: Optional[int],
):
    from omegaconf import OmegaConf

    from data.dataset import create_dataset

    dataset_root_path = Path(dataset_root).resolve()
    config = OmegaConf.load(dataset_config)
    config.common.action_chunk_size = config.common.num_video_frames * config.common.video_action_freq_ratio
    config.dataset.type = "lerobot"
    config.dataset.dataset_dir = str(dataset_root_path)
    config.dataset.task_mode = "single"
    config.dataset.image_aug = False

    if not hasattr(config.dataset, "params") or config.dataset.params is None:
        config.dataset.params = OmegaConf.create({})

    config.dataset.params.repo_id = repo_id or dataset_root_path.name
    config.dataset.params.root = str(dataset_root_path)
    config.dataset.params.embodiment_type = embodiment_type
    config.dataset.params.enable_t5_fallback = bool(enable_t5_fallback)
    if t5_wan_path is not None:
        config.dataset.params.t5_wan_path = t5_wan_path
    if max_episodes is not None:
        config.dataset.max_episodes = max_episodes

    return create_dataset(config, val=False)


def test_basic_connectivity(client: MotusAPIClient) -> bool:
    print("Testing basic connectivity...")
    try:
        health = client.health_check()
        print("Health check passed")
        print(f"  Status: {health.get('status')}")
        print(f"  Model loaded: {health.get('model_loaded')}")
        print(f"  T5 loaded: {health.get('t5_loaded')}")
        print(f"  Device: {health.get('device')}")
        print(f"  Timestamp: {health.get('timestamp')}")
        return True
    except Exception as exc:
        print(f"Health check failed: {exc}")
        return False


def test_model_info(client: MotusAPIClient) -> bool:
    print("\nTesting model info...")
    try:
        info = client.get_model_info()
        print("Model info retrieved")
        print(f"  Device: {info.get('device')}")
        print(f"  Checkpoint: {info.get('checkpoint_path')}")
        print(f"  T5 loaded: {info.get('t5_loaded')}")
        print(f"  T5 embeddings dir: {info.get('t5_embeddings_dir')}")
        common = info.get("common", {})
        print(f"  State dim: {common.get('state_dim')}")
        print(f"  Action dim: {common.get('action_dim')}")
        print(f"  Video size: {common.get('video_height')} x {common.get('video_width')}")
        return True
    except Exception as exc:
        print(f"Model info failed: {exc}")
        return False


def test_real_inference(
    client: MotusAPIClient,
    instruction: str,
    images_b64: Optional[List[str]] = None,
    image_path: Optional[str] = None,
    state: Optional[List[float]] = None,
    t5_embeddings_path: Optional[str] = None,
    t5_embeddings_dir: Optional[str] = None,
    auto_find_t5_embeddings: bool = True,
    num_inference_steps: Optional[int] = None,
    instruction_prefix: Optional[str] = None,
    return_frame_grid: bool = False,
    frame_grid_output: Optional[str] = None,
) -> bool:
    print("\nTesting inference...")
    try:
        if images_b64 is None and image_path is None:
            images_b64 = [
                client.create_random_image_base64(),
                client.create_random_image_base64(),
                client.create_random_image_base64(),
            ]

        start_time = time.time()
        result = client.run_inference(
            instruction=instruction,
            images=images_b64,
            image_path=image_path,
            state=state,
            proprio_data=[state] if state is not None else None,
            t5_embeddings_path=t5_embeddings_path,
            t5_embeddings_dir=t5_embeddings_dir,
            auto_find_t5_embeddings=auto_find_t5_embeddings,
            num_inference_steps=num_inference_steps,
            instruction_prefix=instruction_prefix,
            return_frame_grid=return_frame_grid,
            save_output_path=None,
        )
        elapsed_ms = (time.time() - start_time) * 1000.0

        print(f"Inference completed in {elapsed_ms:.2f}ms")
        print(f"  Effective instruction: {result.get('effective_instruction')}")
        print(f"  Action shape: {result.get('action_shape')}")
        print(f"  Predicted frames shape: {result.get('predicted_frames_shape')}")
        print(f"  Server processing time: {result.get('processing_time_ms', 0):.2f}ms")

        actions = np.array(result.get("predicted_actions", []), dtype=np.float32)
        if actions.size > 0:
            print(f"  First action: {actions[0].tolist()}")

        _save_frame_grid_if_present(result, frame_grid_output)
        if frame_grid_output and result.get("frame_grid_image"):
            print(f"  Saved frame grid to: {frame_grid_output}")
        return True
    except Exception as exc:
        print(f"Inference failed: {exc}")
        return False


def benchmark_inference(
    client: MotusAPIClient,
    instruction: str,
    num_requests: int,
    images_b64: Optional[List[str]] = None,
    image_path: Optional[str] = None,
    state: Optional[List[float]] = None,
    t5_embeddings_path: Optional[str] = None,
    t5_embeddings_dir: Optional[str] = None,
    auto_find_t5_embeddings: bool = True,
    num_inference_steps: Optional[int] = None,
    instruction_prefix: Optional[str] = None,
) -> bool:
    print(f"\nBenchmarking inference ({num_requests} requests)...")
    if images_b64 is None and image_path is None:
        images_b64 = [
            client.create_random_image_base64(),
            client.create_random_image_base64(),
            client.create_random_image_base64(),
        ]

    times: List[float] = []
    for idx in range(num_requests):
        try:
            start_time = time.time()
            client.run_inference(
                instruction=instruction,
                images=images_b64,
                image_path=image_path,
                state=state,
                proprio_data=[state] if state is not None else None,
                t5_embeddings_path=t5_embeddings_path,
                t5_embeddings_dir=t5_embeddings_dir,
                auto_find_t5_embeddings=auto_find_t5_embeddings,
                num_inference_steps=num_inference_steps,
                instruction_prefix=instruction_prefix,
                return_frame_grid=False,
                save_output_path=None,
            )
            request_ms = (time.time() - start_time) * 1000.0
            times.append(request_ms)
            print(f"  Request {idx + 1}/{num_requests}: {request_ms:.2f}ms")
        except Exception as exc:
            print(f"  Request {idx + 1}/{num_requests} failed: {exc}")

    if not times:
        print("No successful benchmark requests")
        return False

    avg_time = float(np.mean(times))
    print("\nBenchmark results:")
    print(f"  Average time: {avg_time:.2f}ms")
    print(f"  Min time: {float(np.min(times)):.2f}ms")
    print(f"  Max time: {float(np.max(times)):.2f}ms")
    print(f"  Std deviation: {float(np.std(times)):.2f}ms")
    print(f"  Throughput: {1000.0 / avg_time:.2f} requests/second")
    return True


def evaluate_lerobot_loss(
    client: MotusAPIClient,
    dataset_root: str,
    dataset_config: str,
    repo_id: Optional[str],
    embodiment_type: str,
    t5_wan_path: Optional[str],
    enable_t5_fallback: bool,
    max_episodes: Optional[int],
    num_samples: int,
    num_inference_steps: Optional[int],
    instruction_prefix: Optional[str],
) -> bool:
    print("\nEvaluating loss with LeRobot dataset via server...")

    try:
        dataset = _load_lerobot_dataset(
            dataset_root=dataset_root,
            dataset_config=dataset_config,
            repo_id=repo_id,
            embodiment_type=embodiment_type,
            t5_wan_path=t5_wan_path,
            enable_t5_fallback=enable_t5_fallback,
            max_episodes=max_episodes,
        )
    except Exception as exc:
        print(f"Failed to load dataset: {exc}")
        traceback.print_exc()
        return False

    mse_losses: List[float] = []
    l1_losses: List[float] = []

    for idx in range(num_samples):
        temp_t5_path = None
        try:
            sample = dataset[idx]
            if sample is None:
                print(f"  Sample {idx + 1}/{num_samples}: skipped (None)")
                continue

            instruction = sample.get("instruction", "")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("Dataset sample does not contain a valid instruction.")

            input_images = sample.get("input_images", None)
            if input_images is not None:
                images_b64 = [_tensor_image_to_base64(image_tensor) for image_tensor in input_images]
            else:
                images_b64 = [_tensor_image_to_base64(sample["first_frame"])]
            state = sample["initial_state"].detach().cpu().float().tolist()
            gt_actions = sample["action_sequence"].detach().cpu().float().numpy()

            language_embedding = sample.get("language_embedding", None)
            if language_embedding is None:
                raise ValueError("Dataset sample does not contain language_embedding.")
            temp_t5_path = _save_temp_t5_embedding(language_embedding)

            start_time = time.time()
            result = client.run_inference(
                instruction=instruction,
                images=images_b64,
                state=state,
                proprio_data=[state],
                t5_embeddings_path=temp_t5_path,
                auto_find_t5_embeddings=False,
                num_inference_steps=num_inference_steps,
                instruction_prefix=instruction_prefix,
                return_frame_grid=False,
                save_output_path=None,
                timeout=300,
            )
            elapsed_ms = (time.time() - start_time) * 1000.0

            pred_actions = np.array(result.get("predicted_actions", []), dtype=np.float32)
            if pred_actions.ndim == 1:
                pred_actions = pred_actions[None, :]
            if gt_actions.ndim == 1:
                gt_actions = gt_actions[None, :]

            seq_len = min(pred_actions.shape[0], gt_actions.shape[0])
            action_dim = min(pred_actions.shape[1], gt_actions.shape[1])
            pred_actions = pred_actions[:seq_len, :action_dim]
            gt_actions = gt_actions[:seq_len, :action_dim]

            mse_loss = float(np.mean((pred_actions - gt_actions) ** 2))
            l1_loss = float(np.mean(np.abs(pred_actions - gt_actions)))
            mse_losses.append(mse_loss)
            l1_losses.append(l1_loss)

            print(
                f"  Sample {idx + 1}/{num_samples}: "
                f"mse={mse_loss:.6f} l1={l1_loss:.6f} "
                f"server_time={result.get('processing_time_ms', 0):.2f}ms "
                f"client_time={elapsed_ms:.2f}ms"
            )
        except Exception as exc:
            print(f"  Sample {idx + 1}/{num_samples} failed: {exc}")
        finally:
            if temp_t5_path and os.path.exists(temp_t5_path):
                os.remove(temp_t5_path)

    if not mse_losses:
        print("No successful LeRobot evaluation samples")
        return False

    print("\nLeRobot loss summary:")
    print(f"  Successful samples: {len(mse_losses)}/{num_samples}")
    print(f"  Average MSE loss: {float(np.mean(mse_losses)):.6f}")
    print(f"  Average L1 loss: {float(np.mean(l1_losses)):.6f}")
    print(f"  MSE std: {float(np.std(mse_losses)):.6f}")
    print(f"  L1 std: {float(np.std(l1_losses)):.6f}")
    return True


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Motus API Client Test Suite")
    parser.add_argument("--url", default="http://localhost:6789", help="API server URL")
    parser.add_argument(
        "--test",
        choices=["connectivity", "model_info", "inference", "benchmark", "lerobot_loss", "all"],
        default="all",
        help="Test to run",
    )
    parser.add_argument("--benchmark_requests", type=int, default=10, help="Number of benchmark requests")
    parser.add_argument("--instruction", type=str, default="cook vegetable", help="Instruction for inference")
    parser.add_argument("--image", type=str, default=None, help="Path to input image file")
    parser.add_argument("--top_image", type=str, default=None, help="Path to top/main camera image file")
    parser.add_argument("--left_wrist_image", type=str, default=None, help="Path to left wrist camera image file")
    parser.add_argument("--right_wrist_image", type=str, default=None, help="Path to right wrist camera image file")
    parser.add_argument("--images", nargs="+", default=None, help="List of image paths, typically 3 images")
    parser.add_argument("--state_csv", type=str, default=None, help="Comma-separated state vector")
    parser.add_argument("--state_json", type=str, default=None, help="Path to a JSON file containing a state vector")
    parser.add_argument("--t5_embeddings_path", type=str, default=None, help="Explicit T5 embedding .pt file path")
    parser.add_argument("--t5_embeddings_dir", type=str, default=None, help="Directory for auto T5 embedding lookup")
    parser.add_argument(
        "--disable_auto_find_t5_embeddings",
        action="store_true",
        help="Disable automatic T5 embedding lookup by instruction",
    )
    parser.add_argument("--num_inference_steps", type=int, default=None, help="Override inference step count")
    parser.add_argument("--instruction_prefix", type=str, default=None, help="Optional instruction prefix")
    parser.add_argument("--return_frame_grid", action="store_true", help="Request frame grid image in response")
    parser.add_argument("--frame_grid_output", type=str, default=None, help="Where to save returned frame grid PNG")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="Dobot/dobot_cook_vegetable_full",
        help="LeRobot dataset root for lerobot_loss mode",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="configs/lerobot.yaml",
        help="Base dataset config for lerobot_loss mode",
    )
    parser.add_argument("--repo_id", type=str, default=None, help="Optional repo id override for lerobot_loss mode")
    parser.add_argument(
        "--embodiment_type",
        type=str,
        default="local",
        help="Normalization embodiment type for lerobot_loss mode",
    )
    parser.add_argument(
        "--t5_wan_path",
        type=str,
        default="pretrained_models",
        help="WAN path used if LeRobot dataset needs on-the-fly T5 fallback",
    )
    parser.add_argument(
        "--disable_dataset_t5_fallback",
        action="store_true",
        help="Disable LeRobot dataset on-the-fly T5 fallback",
    )
    parser.add_argument("--max_episodes", type=int, default=None, help="Optional episode cap for lerobot_loss mode")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of LeRobot samples for loss evaluation")
    return parser


def main() -> int:
    args = build_argparser().parse_args()

    print("Motus API Client Test Suite")
    print("=" * 50)
    print(f"API Server: {args.url}")

    client = MotusAPIClient(args.url)
    tests_passed = 0
    total_tests = 0

    state = None
    if args.state_json:
        state = _read_json_float_list(args.state_json)
    elif args.state_csv:
        state = _parse_csv_to_float_list(args.state_csv)

    images_b64 = None
    image_path = None
    if args.images:
        images_b64 = [client.encode_image_to_base64(path) for path in args.images]
    elif args.top_image or args.left_wrist_image or args.right_wrist_image:
        image_paths = [path for path in [args.top_image, args.left_wrist_image, args.right_wrist_image] if path]
        images_b64 = [client.encode_image_to_base64(path) for path in image_paths]
    elif args.image:
        image_path = args.image

    auto_find_t5_embeddings = not args.disable_auto_find_t5_embeddings
    enable_dataset_t5_fallback = not args.disable_dataset_t5_fallback

    if args.test in ["connectivity", "all"]:
        total_tests += 1
        if test_basic_connectivity(client):
            tests_passed += 1

    if args.test in ["model_info", "all"]:
        total_tests += 1
        if test_model_info(client):
            tests_passed += 1

    if args.test in ["inference", "all"]:
        total_tests += 1
        if test_real_inference(
            client=client,
            images_b64=images_b64,
            instruction=args.instruction,
            image_path=image_path,
            state=state,
            t5_embeddings_path=args.t5_embeddings_path,
            t5_embeddings_dir=args.t5_embeddings_dir,
            auto_find_t5_embeddings=auto_find_t5_embeddings,
            num_inference_steps=args.num_inference_steps,
            instruction_prefix=args.instruction_prefix,
            return_frame_grid=args.return_frame_grid,
            frame_grid_output=args.frame_grid_output,
        ):
            tests_passed += 1

    if args.test in ["benchmark", "all"]:
        total_tests += 1
        if benchmark_inference(
            client=client,
            instruction=args.instruction,
            num_requests=args.benchmark_requests,
            images_b64=images_b64,
            image_path=image_path,
            state=state,
            t5_embeddings_path=args.t5_embeddings_path,
            t5_embeddings_dir=args.t5_embeddings_dir,
            auto_find_t5_embeddings=auto_find_t5_embeddings,
            num_inference_steps=args.num_inference_steps,
            instruction_prefix=args.instruction_prefix,
        ):
            tests_passed += 1

    if args.test == "lerobot_loss":
        total_tests += 1
        if evaluate_lerobot_loss(
            client=client,
            dataset_root=args.dataset_root,
            dataset_config=args.dataset_config,
            repo_id=args.repo_id,
            embodiment_type=args.embodiment_type,
            t5_wan_path=args.t5_wan_path,
            enable_t5_fallback=enable_dataset_t5_fallback,
            max_episodes=args.max_episodes,
            num_samples=args.num_samples,
            num_inference_steps=args.num_inference_steps,
            instruction_prefix=args.instruction_prefix,
        ):
            tests_passed += 1

    print("\n" + "=" * 50)
    print("Test Summary")
    print(f"Tests passed: {tests_passed}/{total_tests}")

    return 0 if tests_passed == total_tests else 1


if __name__ == "__main__":
    raise SystemExit(main())

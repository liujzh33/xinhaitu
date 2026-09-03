#!/usr/bin/env python3
"""
WebSocket inference server for MotusWanVlmDirectMask on RoboTwin.

This server loads the model, receives observations from the client (RoboTwin env),
runs inference, and returns predicted actions.

Usage (terminal 1):
    python dobot_websocket_vlm_server.py \
      --model_config /path/to/robotwin_wan_vlm.yml \
      --ckpt_dir /path/to/checkpoint/pytorch_model \
      --wan_path /path/to/Wan2.2-TI2V-5B \
      --vlm_path /path/to/Qwen3-VL-2B-Instruct \
      --device cuda:0 \
      --port 6790
"""

import argparse
import asyncio
import base64
import hashlib
import http
import io
import json
import logging
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import msgpack
import numpy as np
import torch
import websockets
import websockets.asyncio.server as ws_server
import websockets.frames
from PIL import Image
from transformers import AutoProcessor

PROJ_ROOT = str(Path(__file__).resolve().parents[2])
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

# Add RobboTwin_test paths for local model and utility imports
ROBBOTWIN_POLICY_DIR = str(Path(__file__).resolve().parent.parent / "RobboTwin_test" / "policy" / "MotusWanVlmDirectMask")
if ROBBOTWIN_POLICY_DIR not in sys.path:
    sys.path.insert(0, ROBBOTWIN_POLICY_DIR)
ROBBOTWIN_MODELS_DIR = os.path.join(ROBBOTWIN_POLICY_DIR, "models")
if ROBBOTWIN_MODELS_DIR not in sys.path:
    sys.path.insert(0, ROBBOTWIN_MODELS_DIR)
ROBBOTWIN_UTILS_DIR = os.path.join(ROBBOTWIN_POLICY_DIR, "utils")
if ROBBOTWIN_UTILS_DIR not in sys.path:
    sys.path.insert(0, ROBBOTWIN_UTILS_DIR)
BAK_ROOT = str(Path(__file__).resolve().parent.parent / "RobboTwin_test" / "policy" / "MotusWanVlmDirectMask" / "bak")
if BAK_ROOT not in sys.path:
    sys.path.insert(0, BAK_ROOT)

# Import from RobboTwin_test copy (avoids decord dependency in main data.utils.image_utils)
from utils.image_utils import resize_with_padding
from models.motus_wan_vlm_direct_mask import MotusWanVlmDirectMask, MotusWanVlmDirectMaskConfig
from wan.modules.t5 import T5EncoderModel


log = logging.getLogger("dobot_websocket_vlm_server")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Scene prefix used in deploy_policy.py
SCENE_PREFIX = (
    "The whole scene is in a realistic, industrial art style with three views: "
    "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
    "The aloha robot is currently performing the following task: "
)

# Action normalization stats from stat.json (robotwin2)
ACTION_STATS = {
    "min": np.array([
        -1.4206113815307617, -0.005189999938011169, -0.18577136099338531,
        -1.956291913986206, -1.6801013946533203, -3.965158462524414, 0.0,
        -2.9455959796905518, -0.6294453740119934, -0.07775841653347015,
        -2.0402400493621826, -2.1285502910614014, -3.9987125396728516, 0.0,
    ], dtype=np.float32),
    "max": np.array([
        0.5171146392822266, 3.4578874111175537, 3.6927058696746826,
        1.7942548990249634, 1.8604620695118136, 3.989658832550049, 1.0,
        1.4472923278808594, 3.640752077102661, 3.9892151355748341,
        1.95806884765625, 1.5265172719955444, 3.9609365463256836, 1.0,
    ], dtype=np.float32),
}


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


class ServerState:
    def __init__(self) -> None:
        self.model: Optional[MotusWanVlmDirectMask] = None
        self.processor: Optional[AutoProcessor] = None
        self.t5_encoder: Optional[T5EncoderModel] = None
        self.config_dict: Optional[Dict[str, Any]] = None
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.default_steps: int = 10
        self.lock = threading.Lock()
        self.checkpoint_path: Optional[str] = None
        self.wan_path: Optional[str] = None
        self.vlm_path: Optional[str] = None
        self.action_min: Optional[np.ndarray] = None
        self.action_max: Optional[np.ndarray] = None
        self.action_range: Optional[np.ndarray] = None


SERVER_STATE = ServerState()


def load_yaml_config(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def decode_base64_image(image_base64: str) -> Image.Image:
    if image_base64.startswith("data:image"):
        image_base64 = image_base64.split(",", 1)[1]
    image_bytes = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def image_array_to_pil(image_array: np.ndarray) -> Image.Image:
    array = np.asarray(image_array)
    if array.ndim != 3:
        raise ValueError(f"Expected image array with 3 dims, got shape {array.shape}.")
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array, 0.0, 1.0) * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return Image.fromarray(array[..., :3]).convert("RGB")


def any_image_to_pil(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, np.ndarray):
        return image_array_to_pil(value)
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, str):
        maybe_path = Path(value)
        if maybe_path.exists():
            return Image.open(maybe_path).convert("RGB")
        return decode_base64_image(value)
    raise ValueError(f"Unsupported image value type: {type(value)!r}")


def resize_image_with_padding(image: Image.Image, size_hw: tuple[int, int]) -> Image.Image:
    image_np = np.array(image).astype(np.uint8)
    resized_np = resize_with_padding(image_np, size_hw)
    return Image.fromarray(resized_np)


def image_to_tensor(image: Image.Image, size_hw: tuple[int, int]) -> torch.Tensor:
    image = resize_image_with_padding(image, size_hw)
    image_np = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)


def build_vlm_inputs(
    processor: AutoProcessor,
    instruction: str,
    image: Image.Image,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": image},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

    encoded = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    )

    vlm_inputs: Dict[str, torch.Tensor] = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "pixel_values": encoded["pixel_values"].to(device),
    }
    if encoded.get("image_grid_thw") is not None:
        vlm_inputs["image_grid_thw"] = encoded["image_grid_thw"].to(device)
    return vlm_inputs


def resolve_wan_model_dir(path: str) -> str:
    raw_path = Path(path).expanduser().resolve()
    if (raw_path / "Wan2.2_VAE.pth").exists():
        return str(raw_path)
    nested_path = raw_path / "Wan2.2-TI2V-5B"
    if (nested_path / "Wan2.2_VAE.pth").exists():
        return str(nested_path)
    raise ValueError(f"Cannot resolve WAN model directory from {path}")


def load_server_components(
    model_config_path: str,
    ckpt_dir: str,
    wan_path: str,
    vlm_path: Optional[str],
    device: Optional[str],
) -> None:
    resolved_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    config_dict = load_yaml_config(model_config_path)

    resolved_wan_path = resolve_wan_model_dir(wan_path)
    resolved_vlm_path = (
        str(Path(vlm_path).expanduser().resolve())
        if vlm_path
        else config_dict["model"]["vlm"]["checkpoint_path"]
    )

    common = config_dict["common"]
    model_cfg = config_dict["model"]

    def to_float(val: Any) -> Any:
        return float(val) if isinstance(val, str) else val

    model_config = MotusWanVlmDirectMaskConfig(
        wan_checkpoint_path=resolved_wan_path,
        vae_path=os.path.join(resolved_wan_path, "Wan2.2_VAE.pth"),
        wan_config_path=resolved_wan_path,
        vlm_checkpoint_path=resolved_vlm_path,
        video_precision="bfloat16",
        action_state_dim=common["state_dim"],
        action_dim=common["action_dim"],
        action_expert_dim=model_cfg["action_expert"]["hidden_size"],
        action_expert_ffn_dim_multiplier=model_cfg["action_expert"]["ffn_dim_multiplier"],
        action_expert_norm_eps=to_float(model_cfg["action_expert"].get("norm_eps", 1e-6)),
        vlm_dim=model_cfg["qwen3_expert"]["vlm_dim"],
        qwen3_expert_head_dim=model_cfg["qwen3_expert"]["head_dim"],
        qwen3_expert_num_heads=model_cfg["qwen3_expert"]["num_heads"],
        qwen3_expert_num_layers=model_cfg["qwen3_expert"]["num_layers"],
        qwen3_expert_norm_eps=to_float(model_cfg["qwen3_expert"].get("norm_eps", 1e-5)),
        global_downsample_rate=common["global_downsample_rate"],
        video_action_freq_ratio=common["video_action_freq_ratio"],
        num_video_frames=common["num_video_frames"],
        video_height=common["video_height"],
        video_width=common["video_width"],
        batch_size=1,
        video_loss_weight=model_cfg["loss_weights"]["video_loss_weight"],
        action_loss_weight=model_cfg["loss_weights"]["action_loss_weight"],
        training_mode="finetune",
        load_pretrained_backbones=False,
        vlm_frozen=model_cfg.get("vlm", {}).get("frozen", False),
        subtask_prediction=model_cfg.get("subtask_prediction"),
        progress_detection=model_cfg.get("progress_detection"),
    )

    model = MotusWanVlmDirectMask(model_config).to(resolved_device)
    model.load_checkpoint(ckpt_dir, strict=False)
    model.eval()
    model.vlm_model = model.vlm_model.to(resolved_device)
    processor = AutoProcessor.from_pretrained(resolved_vlm_path, trust_remote_code=True)

    # Initialize T5 encoder for on-the-fly instruction encoding
    t5_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=str(resolved_device),
        checkpoint_path=os.path.join(resolved_wan_path, "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=os.path.join(resolved_wan_path, "google", "umt5-xxl"),
    )

    # Load action normalization stats
    action_min = ACTION_STATS["min"]
    action_max = ACTION_STATS["max"]
    action_range = action_max - action_min

    SERVER_STATE.model = model
    SERVER_STATE.processor = processor
    SERVER_STATE.t5_encoder = t5_encoder
    SERVER_STATE.config_dict = config_dict
    SERVER_STATE.device = resolved_device
    SERVER_STATE.default_steps = int(model_cfg.get("inference", {}).get("num_inference_timesteps", 10))
    SERVER_STATE.checkpoint_path = ckpt_dir
    SERVER_STATE.wan_path = resolved_wan_path
    SERVER_STATE.vlm_path = resolved_vlm_path
    SERVER_STATE.action_min = action_min
    SERVER_STATE.action_max = action_max
    SERVER_STATE.action_range = action_range

    log.info("Loaded model from %s on %s", ckpt_dir, resolved_device)


def get_input_image(obs: Dict[str, Any]) -> Image.Image:
    """Get composed image from observation (supports both dict and list formats)."""
    if "images" in obs and obs["images"] is not None:
        images_value = obs["images"]
        if isinstance(images_value, dict):
            ordered = [
                images_value[key]
                for key in ("top", "left_wrist", "right_wrist")
                if key in images_value and images_value[key] is not None
            ]
            if not ordered:
                ordered = [value for value in images_value.values() if value is not None]
            images = [any_image_to_pil(value) for value in ordered]
        else:
            images = [any_image_to_pil(value) for value in images_value]
        return compose_multiview_image(images)

    if obs.get("image") is not None:
        return any_image_to_pil(obs["image"])

    if obs.get("image_path") is not None:
        return Image.open(obs["image_path"]).convert("RGB")

    raise ValueError("Observation must contain 'images', 'image', or 'image_path'.")


def compose_multiview_image(images: List[Image.Image]) -> Image.Image:
    if len(images) == 0:
        raise ValueError("At least one image is required.")
    if len(images) == 1:
        return images[0].convert("RGB")

    top = np.array(images[0].convert("RGB")).astype(np.uint8)
    top_h, top_w = top.shape[:2]
    bottom_h = max(1, top_h // 2)
    left_w = max(1, top_w // 2)
    right_w = top_w - left_w

    left_img = np.array(images[1].convert("RGB")).astype(np.uint8)
    right_img = left_img if len(images) == 2 else np.array(images[2].convert("RGB")).astype(np.uint8)

    left_resized = np.array(Image.fromarray(left_img).resize((left_w, bottom_h), Image.BICUBIC))
    right_resized = np.array(Image.fromarray(right_img).resize((right_w, bottom_h), Image.BICUBIC))
    bottom_row = np.concatenate([left_resized, right_resized], axis=1)
    composed = np.concatenate([top, bottom_row], axis=0)
    return Image.fromarray(composed)


def get_state_tensor(state_values: Any, state_dim: int, device: torch.device) -> torch.Tensor:
    if state_values is None:
        return torch.zeros((1, state_dim), dtype=torch.float32, device=device)

    state_array = np.asarray(state_values, dtype=np.float32).reshape(-1)
    if state_array.shape[0] != state_dim:
        raise ValueError(f"State length mismatch: expected {state_dim}, got {state_array.shape[0]}.")
    return torch.from_numpy(state_array).to(device=device, dtype=torch.float32).unsqueeze(0)


def denormalize_actions(actions: np.ndarray) -> np.ndarray:
    """Denormalize actions from [0,1] to real range using stat.json stats."""
    action_min = SERVER_STATE.action_min
    action_max = SERVER_STATE.action_max
    action_range = SERVER_STATE.action_range
    return actions * action_range + action_min


def run_model_inference(obs: Dict[str, Any]) -> Dict[str, Any]:
    if SERVER_STATE.model is None or SERVER_STATE.processor is None or SERVER_STATE.config_dict is None:
        raise RuntimeError("Model is not loaded.")

    instruction = str(obs.get("instruction", "")).strip()
    if not instruction:
        raise ValueError("Observation field 'instruction' is required.")

    cfg = SERVER_STATE.config_dict
    common = cfg["common"]
    num_inference_steps = int(obs.get("num_inference_steps") or SERVER_STATE.default_steps)

    # Add scene prefix (matching deploy_policy.py)
    effective_instruction = f"{SCENE_PREFIX}{instruction}"

    start_time = datetime.now()

    # Prepare image
    input_image = get_input_image(obs)
    image_hw = (int(common["video_height"]), int(common["video_width"]))
    resized_pil = resize_image_with_padding(input_image, image_hw)
    first_frame = image_to_tensor(input_image, image_hw).to(SERVER_STATE.device)

    # Prepare state
    state_dim = int(common["state_dim"])
    state = get_state_tensor(obs.get("state"), state_dim, SERVER_STATE.device)

    # Prepare VLM inputs
    vlm_inputs = [build_vlm_inputs(SERVER_STATE.processor, effective_instruction, resized_pil, SERVER_STATE.device)]

    # Encode instruction with T5 (on-the-fly, same as deploy_policy.py)
    t5_out = SERVER_STATE.t5_encoder([effective_instruction], SERVER_STATE.device)
    if isinstance(t5_out, torch.Tensor):
        language_embeddings = [t5_out.squeeze(0)] if t5_out.dim() == 3 else [t5_out]
    elif isinstance(t5_out, list):
        language_embeddings = t5_out
    else:
        raise ValueError("Unexpected T5 encoder output format")

    # Run inference (returns 4 values: frames, actions, progress, subtask)
    with SERVER_STATE.lock:
        with torch.inference_mode():
            predicted_frames, predicted_actions, predicted_progress, predicted_subtask = (
                SERVER_STATE.model.inference_step(
                    first_frame=first_frame,
                    state=state,
                    num_inference_steps=num_inference_steps,
                    language_embeddings=language_embeddings,
                    vlm_inputs=vlm_inputs,
                )
            )

    end_time = datetime.now()
    predicted_actions_cpu = predicted_actions.detach().cpu().float()
    if predicted_actions_cpu.dim() == 3:
        predicted_actions_cpu = predicted_actions_cpu.squeeze(0)
    if predicted_actions_cpu.dim() == 1:
        predicted_actions_cpu = predicted_actions_cpu.unsqueeze(0)

    # Model outputs actions in real range directly (no denormalization needed)
    predicted_actions_np = predicted_actions_cpu.numpy()

    result: Dict[str, Any] = {
        "instruction": instruction,
        "effective_instruction": effective_instruction,
        "predicted_actions": predicted_actions_np,
        "action_shape": list(predicted_actions_np.shape),
        "predicted_frames_shape": list(predicted_frames.shape),
        "processing_time_ms": (end_time - start_time).total_seconds() * 1000.0,
        "model_info": {
            "device": str(SERVER_STATE.device),
            "num_inference_steps": num_inference_steps,
            "state_dim": state_dim,
            "action_dim": int(common["action_dim"]),
            "video_size": [int(common["video_height"]), int(common["video_width"])],
            "num_video_frames": int(common["num_video_frames"]),
        },
        "timestamp": end_time.isoformat(),
    }

    # Include subtask and progress predictions if available
    if predicted_progress is not None:
        result["predicted_progress"] = predicted_progress.detach().cpu().float().item()
    if predicted_subtask is not None:
        result["predicted_subtask"] = str(predicted_subtask)

    return result


def server_metadata() -> Dict[str, Any]:
    common = SERVER_STATE.config_dict["common"] if SERVER_STATE.config_dict else {}
    return {
        "server": "dobot_websocket_vlm_server",
        "protocol": "websocket-msgpack-numpy",
        "model_loaded": SERVER_STATE.model is not None,
        "device": str(SERVER_STATE.device),
        "checkpoint_path": SERVER_STATE.checkpoint_path,
        "wan_path": SERVER_STATE.wan_path,
        "vlm_path": SERVER_STATE.vlm_path,
        "common": {
            "state_dim": int(common["state_dim"]) if "state_dim" in common else None,
            "action_dim": int(common["action_dim"]) if "action_dim" in common else None,
            "video_height": int(common["video_height"]) if "video_height" in common else None,
            "video_width": int(common["video_width"]) if "video_width" in common else None,
            "num_video_frames": int(common["num_video_frames"]) if "num_video_frames" in common else None,
            "video_action_freq_ratio": int(common["video_action_freq_ratio"]) if "video_action_freq_ratio" in common else None,
        },
    }


async def websocket_handler(websocket: ws_server.ServerConnection) -> None:
    log.info("Connection from %s opened", websocket.remote_address)
    await websocket.send(pack_message(server_metadata()))

    prev_total_time = None
    while True:
        try:
            start_time = time.monotonic()
            raw_message = await websocket.recv()
            if isinstance(raw_message, str):
                raise ValueError("Expected binary MessagePack request, got text frame.")

            obs = unpack_message(raw_message)
            infer_start = time.monotonic()
            result = run_model_inference(obs)
            infer_ms = (time.monotonic() - infer_start) * 1000.0
            result["server_timing"] = {"infer_ms": infer_ms}
            if prev_total_time is not None:
                result["server_timing"]["prev_total_ms"] = prev_total_time * 1000.0

            await websocket.send(pack_message(result))
            prev_total_time = time.monotonic() - start_time

        except websockets.ConnectionClosed:
            log.info("Connection from %s closed", websocket.remote_address)
            break
        except Exception:
            error_text = traceback.format_exc()
            log.error("Inference request failed:\n%s", error_text)
            await websocket.send(error_text)
            await websocket.close(
                code=websockets.frames.CloseCode.INTERNAL_ERROR,
                reason="Internal server error. Traceback included in previous frame.",
            )
            break


def health_check(connection: ws_server.ServerConnection, request: ws_server.Request) -> ws_server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


async def run_websocket_server(host: str, port: int) -> None:
    async with ws_server.serve(
        websocket_handler,
        host,
        port,
        compression=None,
        max_size=None,
        process_request=health_check,
    ) as server:
        log.info("WebSocket inference server listening on ws://%s:%s", host, port)
        await server.serve_forever()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MotusWanVlmDirectMask WebSocket inference server for RoboTwin")
    parser.add_argument("--model_config", required=True, help="Path to YAML config (robotwin_wan_vlm.yml)")
    parser.add_argument("--ckpt_dir", required=True, help="Path to checkpoint directory")
    parser.add_argument("--wan_path", required=True, help="WAN model path")
    parser.add_argument("--vlm_path", default=None, help="VLM path override")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=6790, type=int)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = build_argparser().parse_args()

    load_server_components(
        model_config_path=args.model_config,
        ckpt_dir=args.ckpt_dir,
        wan_path=args.wan_path,
        vlm_path=args.vlm_path,
        device=args.device,
    )

    asyncio.run(run_websocket_server(args.host, args.port))


if __name__ == "__main__":
    main()

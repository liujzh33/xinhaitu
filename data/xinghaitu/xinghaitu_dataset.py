"""X1-Robot (华为 HDAS) Dataset Loader for Motus

数据格式 (Motus 转换后):
  dataset_dir/
  ├── episode_000000/
  │   ├── videos/episode_000000.mp4    # T 型拼接 320×360
  │   ├── qpos/episode_000000.pt       # [T, 40] = [state(24), action(16)]
  │   ├── metas/episode_000000.txt      # meta_prefix + instruction
  │   └── umt5_wan/episode_000000.pt   # T5 embedding [1, S, D]
  ├── episode_000001/
  │   └── ...
  └── readme.md

qpos 维度布局 (40 维):
  [0:6]   base (steer×3 + wheel×3) — 全为 0
  [6:10]  torso (4 关节)
  [10:17] left_arm (7 关节)
  [17:24] right_arm (7 关节)
  [24:31] action.left_arm (7 关节)
  [31:38] action.right_arm (7 关节)
  [38]    action.left_gripper
  [39]    action.right_gripper

可通过 action_slice / state_slice 配置选择使用哪些维度。
"""
import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as data
import warnings

from data.utils.image_utils import load_video_frames, get_video_frame_count, tensor_to_pil
from data.utils.norm import normalize_actions, load_normalization_stats
from utils.vlm_utils import preprocess_vlm_messages

try:
    from transformers import AutoProcessor
except Exception:
    AutoProcessor = None

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


def _clean_subtask_text(text: str) -> str:
    """Clean subtask text: remove role markers, fix punctuation, strip whitespace."""
    text = text.strip()
    # Remove bracketed markers like [Subtask]
    text = text.replace("[Subtask]", "").strip()
    # Remove "Subtask:" prefix if present
    if text.lower().startswith("subtask:"):
        text = text.split(":", 1)[1].strip()
    # Fix double/multiple periods
    while ".." in text:
        text = text.replace("..", ".")
    # Remove chat role tokens that should never appear
    text = text.replace("assistant", "").replace("user", "").strip()
    return text


class XinghaituMotusDataset(data.Dataset):
    """Motus 格式的 X1-Robot 数据集加载器。

    与 RobotWinTaskDataset 接口对齐,但:
    - 目录结构为 episode_XXXXXX/{videos,qpos,metas,umt5_wan}/
    - qpos 为 [T, 40] = [state(24), action(16)], 可通过 qpos_indices 配置选择维度
    - 无 task 子目录 (单一 task: huawei_hdas_teleop)

    qpos 维度布局 (40 维):
      [0:6]   base (steer×3 + wheel×3) — 全为 0
      [6:10]  torso (4 关节)
      [10:17] left_arm (7 关节, state)
      [17:24] right_arm (7 关节, state)
      [24:31] action.left_arm (7 关节, = state[10:17])
      [31:38] action.right_arm (7 关节, = state[17:24])
      [38]    action.left_gripper
      [39]    action.right_gripper

    默认输出 20 维 = torso(4) + left_arm(7) + right_arm(7) + left_gripper(1) + right_gripper(1)
    = qpos[:, [6,7,8,9, 24,25,26,27,28,29,30, 31,32,33,34,35,36,37, 38,39]]
    """

    def __init__(
        self,
        dataset_dir: str,
        # 采样参数
        global_downsample_rate: int = 1,
        video_action_freq_ratio: int = 2,
        num_video_frames: int = 8,
        video_size: Tuple[int, int] = (384, 320),  # (H, W)

        # Episode 限制
        max_episodes: Optional[int] = None,
        val: bool = False,
        image_aug: bool = False,

        # VLM
        vlm_checkpoint_path: Optional[str] = None,

        # qpos 维度选择
        # 默认: 20 维 = torso(4) + left_arm(7) + right_arm(7) + left_gripper(1) + right_gripper(1)
        # qpos 索引: [6,7,8,9] (torso) + [24..30] (left_arm from action) + [31..37] (right_arm from action) + [38] (l_grip) + [39] (r_grip)
        qpos_indices: Optional[List[int]] = None,  # 若为 None 则用默认 20 维

        # 归一化
        embodiment_type: str = "xinghaitu",  # stat.json 中的 key

        **kwargs,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.global_downsample_rate = global_downsample_rate
        self.video_action_freq_ratio = video_action_freq_ratio
        self.num_video_frames = num_video_frames
        self.video_size = video_size
        self.action_chunk_size = num_video_frames * video_action_freq_ratio
        self.max_episodes = max_episodes
        self.val = val
        self.image_aug = image_aug
        self.embodiment_type = embodiment_type

        # 默认 qpos_indices: 20 维, 前 16 维与预训练对齐
        # 预训练 16 维布局: [left_arm(6), pad(1), L_gripper(1), right_arm(6), pad(1), R_gripper(1)]
        # X1-Robot 20 维布局:
        #   [0:7]   left_arm(7)     ← qpos[24:31]  (比预训练多1维, 第7维填入预训练的pad位)
        #   [7]     L_gripper(1)   ← qpos[38]
        #   [8:15]  right_arm(7)    ← qpos[31:38]  (比预训练多1维, 第7维填入预训练的pad位)
        #   [15]    R_gripper(1)   ← qpos[39]
        #   [16:20] torso(4)        ← qpos[6:10]   (新增维度, 放最后)
        if qpos_indices is None:
            self.qpos_indices = (
                list(range(24, 31)) +  # [0:7]  left_arm
                [38] +                  # [7]    L_gripper
                list(range(31, 38)) +  # [8:15] right_arm
                [39] +                  # [15]   R_gripper
                list(range(6, 10))      # [16:20] torso
            )
        else:
            self.qpos_indices = qpos_indices
        self.action_dim = len(self.qpos_indices)

        # VLM processor
        self.vlm_processor = None
        if vlm_checkpoint_path:
            if AutoProcessor is None:
                logger.warning("transformers not installed, VLM processing disabled")
            else:
                try:
                    self.vlm_processor = AutoProcessor.from_pretrained(vlm_checkpoint_path)
                    logger.info(f"VLM processor loaded from {vlm_checkpoint_path}")
                except Exception as e:
                    logger.warning(f"Failed to load VLM processor: {e}")

        # 扫描 episode
        self.episode_files = self._scan_episodes()
        self.total_episodes = len(self.episode_files)

        logger.info(f"XinghaituMotusDataset initialized:")
        logger.info(f"  dataset_dir: {self.dataset_dir}")
        logger.info(f"  total_episodes: {self.total_episodes}")
        logger.info(f"  qpos_indices: {self.qpos_indices}")
        logger.info(f"  action_dim: {self.action_dim}")
        logger.info(f"  global_downsample_rate: {global_downsample_rate}")
        logger.info(f"  video_action_freq_ratio: {video_action_freq_ratio}")
        logger.info(f"  action_chunk_size: {self.action_chunk_size}")
        logger.info(f"  num_video_frames: {num_video_frames}")

    def _scan_episodes(self) -> List[Dict[str, Any]]:
        """扫描 dataset_dir 下的 episode_XXXXXX 目录"""
        episodes = []
        for ep_dir in sorted(self.dataset_dir.iterdir()):
            if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
                continue
            ep_name = ep_dir.name  # episode_000000
            qpos_path = ep_dir / "qpos" / f"{ep_name}.pt"
            video_path = ep_dir / "videos" / f"{ep_name}.mp4"
            lang_path = ep_dir / "umt5_wan" / f"{ep_name}.pt"
            meta_path = ep_dir / "metas" / f"{ep_name}.txt"
            phase_path = ep_dir / "phase_info" / f"{ep_name}.json"

            if not all([qpos_path.exists(), video_path.exists(), lang_path.exists()]):
                logger.warning(f"  {ep_name}: 缺少文件 (qpos={qpos_path.exists()}, video={video_path.exists()}, t5={lang_path.exists()})")
                continue
            episodes.append({
                "episode_name": ep_name,
                "qpos_path": str(qpos_path),
                "video_path": str(video_path),
                "lang_path": str(lang_path),
                "meta_path": str(meta_path),
                # 子任务标注 (可选, X1-Robot-motus-subtask 才有; 普通版为 None)
                "phase_path": str(phase_path) if phase_path.exists() else None,
            })

        if self.max_episodes is not None and self.max_episodes > 0:
            episodes = episodes[:min(self.max_episodes, len(episodes))]

        if len(episodes) == 0:
            raise ValueError(f"No valid episodes found in {self.dataset_dir}")
        return episodes

    def _calculate_sampling_indices(self, total_frames: int) -> Tuple[int, List[int], List[int]]:
        """计算采样索引 (与 robotwin 相同的逻辑)"""
        physical_chunk_size = self.action_chunk_size * self.global_downsample_rate
        max_condition_idx = total_frames - physical_chunk_size - 1

        if max_condition_idx < 0:
            condition_frame_idx = 0
        else:
            condition_frame_idx = random.randint(0, max_condition_idx)

        action_indices = []
        for i in range(self.action_chunk_size):
            action_idx = condition_frame_idx + (i + 1) * self.global_downsample_rate
            action_indices.append(min(action_idx, total_frames - 1))

        video_indices = []
        for i in range(self.num_video_frames):
            action_step = (i + 1) * self.video_action_freq_ratio - 1
            if action_step < len(action_indices):
                video_indices.append(action_indices[action_step])
            else:
                video_indices.append(action_indices[-1])

        return condition_frame_idx, video_indices, action_indices

    def _load_robot_data(self, qpos_path: str, action_indices: List[int],
                         initial_state_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """加载 qpos 并提取 state 和 action

        qpos: [T, 40] = [state(24), action(16)]
        通过 qpos_indices 选择维度, 默认 20 维:
          torso(4) + left_arm(7) + right_arm(7) + left_gripper(1) + right_gripper(1)
          = qpos[:, [6,7,8,9, 24,25,...,39]]

        state 和 action 使用相同的维度选择 (abs 位置控制, state=当前帧, action=目标帧)
        """
        qpos_data = torch.load(qpos_path, map_location='cpu')  # [T, 40]

        if initial_state_idx >= len(qpos_data):
            initial_state_idx = len(qpos_data) - 1

        # 用 qpos_indices 提取指定维度
        indices = torch.tensor(self.qpos_indices, dtype=torch.long)
        initial_state = qpos_data[initial_state_idx, indices].float()

        actions = []
        for idx in action_indices:
            if idx >= len(qpos_data):
                idx = len(qpos_data) - 1
            actions.append(qpos_data[idx, indices])
        action_sequence = torch.stack(actions).float()

        return initial_state, action_sequence

    def _load_language_embedding(self, lang_path: str) -> Tuple[torch.Tensor, int]:
        """加载 T5 embedding (与 robotwin 相同)"""
        embedding_data = torch.load(lang_path, map_location='cpu')
        # X1-Robot 的 umt5_wan 格式: [emb] (list with 1 element)
        if isinstance(embedding_data, list):
            selected_idx = random.randint(0, len(embedding_data) - 1)
            embeddings = embedding_data[selected_idx]
        else:
            selected_idx = 0
            embeddings = embedding_data

        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(0)

        return embeddings, selected_idx

    def _load_text_instruction(self, meta_path: str) -> str:
        """加载 meta 文本 (meta_prefix + instruction)"""
        with open(meta_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        return text

    def __len__(self) -> int:
        return self.total_episodes * 10

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        max_attempts = 8
        for _ in range(max_attempts):
            episode_data = random.choice(self.episode_files)
            try:
                total_frames = get_video_frame_count(episode_data['video_path'])
                if total_frames < 2:
                    continue

                condition_frame_idx, video_indices, action_indices = self._calculate_sampling_indices(total_frames)

                first_frame = load_video_frames(episode_data['video_path'], [condition_frame_idx], self.video_size)
                video_frames = load_video_frames(episode_data['video_path'], video_indices, self.video_size)
                initial_state, action_sequence = self._load_robot_data(
                    episode_data['qpos_path'], action_indices, condition_frame_idx
                )
                language_embedding, instruction_idx = self._load_language_embedding(episode_data['lang_path'])

                text_instruction = self._load_text_instruction(episode_data['meta_path'])

                vlm_inputs = None
                if self.vlm_processor is not None:
                    first_frame_pil = tensor_to_pil(first_frame.squeeze(0))
                    vlm_inputs = preprocess_vlm_messages(text_instruction, first_frame_pil, self.vlm_processor)

                # ===== 子任务 / 进度标注 (可选: phase_info 存在才启用) =====
                subtask_prompt = None
                subtask_lm_inputs = None
                subtask_dec_input_ids = None
                subtask_dec_labels = None
                progress_target = None
                phase_path = episode_data.get('phase_path')
                if phase_path is not None:
                    try:
                        with open(phase_path, 'r', encoding='utf-8') as f:
                            phase_data = json.load(f)
                        phase_info_dict = phase_data.get('phase_info', {})
                        subtasks_list = phase_data.get('subtasks', [])

                        # 按 condition_frame_idx 定位当前子任务 (checkpoints 为各子任务结束帧)
                        checkpoints = phase_info_dict.get('checkpoints', [])
                        phase_idx = 0
                        for i, cp in enumerate(checkpoints):
                            if condition_frame_idx < cp:
                                phase_idx = i
                                break
                        else:
                            phase_idx = len(checkpoints)

                        # 取当前子任务文本 (单 instruction, subtasks[0])
                        if subtasks_list and len(subtasks_list) > 0:
                            task_subtasks = subtasks_list[0]
                            if isinstance(task_subtasks, list) and len(task_subtasks) > 0:
                                phase_idx_clamped = min(phase_idx, len(task_subtasks) - 1)
                                subtask_text = task_subtasks[phase_idx_clamped]
                                target_text_clean = _clean_subtask_text(subtask_text)
                                subtask_prompt = target_text_clean

                                if self.vlm_processor is not None:
                                    try:
                                        target_text = _clean_subtask_text(subtask_text)
                                        if target_text:
                                            max_subtask_len = 64
                                            tokenizer = self.vlm_processor.tokenizer

                                            # --- mot_decoder 格式 ---
                                            target_ids_full = tokenizer(
                                                target_text, add_special_tokens=False
                                            )['input_ids']
                                            if len(target_ids_full) > max_subtask_len:
                                                logger.warning(
                                                    f"[SUBTASK] Text exceeds max_length ({len(target_ids_full)} > {max_subtask_len}), "
                                                    f"truncating: '{target_text[:100]}'"
                                                )
                                            target_ids = target_ids_full[:max_subtask_len]

                                            bos_id = tokenizer.bos_token_id
                                            if bos_id is None:
                                                bos_id = tokenizer.eos_token_id

                                            decoder_input_ids = [bos_id] + target_ids[:-1]
                                            decoder_labels = list(target_ids)

                                            pad_id = tokenizer.pad_token_id or 0
                                            pad_len = max_subtask_len - len(decoder_input_ids)
                                            if pad_len > 0:
                                                decoder_input_ids = decoder_input_ids + [pad_id] * pad_len
                                                decoder_labels = decoder_labels + [-100] * pad_len
                                            else:
                                                decoder_input_ids = decoder_input_ids[:max_subtask_len]
                                                decoder_labels = decoder_labels[:max_subtask_len]

                                            subtask_dec_input_ids = torch.tensor([decoder_input_ids], dtype=torch.long)
                                            subtask_dec_labels = torch.tensor([decoder_labels], dtype=torch.long)

                                            # --- native_lm 格式 (legacy, 兼容 mode=native_lm) ---
                                            clean_instruction = text_instruction.replace(
                                                "The whole scene is in a realistic, industrial art style with three views: "
                                                "a fixed front camera, a movable left hand camera, and a movable right hand camera. "
                                                "The huawei hdas dual-arm robot is currently performing the following task: ", ""
                                            ).strip()
                                            full_text = f"Task: {clean_instruction}\nSubtask: {target_text}"
                                            full_ids = tokenizer(full_text, add_special_tokens=False)['input_ids']

                                            prefix_len = None
                                            for i in range(len(full_ids)):
                                                decoded = tokenizer.decode(full_ids[i:], skip_special_tokens=True).strip()
                                                if decoded == target_text:
                                                    prefix_len = i
                                                    break
                                            if prefix_len is None:
                                                prefix_len = len(full_ids) - 1

                                            img_pad_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
                                            orig_ids = vlm_inputs['input_ids'][0] if vlm_inputs is not None else torch.tensor([])
                                            num_image_tokens = (orig_ids == img_pad_id).sum().item() if vlm_inputs is not None else 0

                                            image_pad_tokens = torch.full((1, num_image_tokens), img_pad_id, dtype=torch.long)
                                            full_ids_tensor = torch.tensor([full_ids], dtype=torch.long)
                                            nlm_input_ids = torch.cat([image_pad_tokens, full_ids_tensor], dim=1)
                                            nlm_attention_mask = torch.ones_like(nlm_input_ids)

                                            nlm_labels = torch.full_like(nlm_input_ids[0], -100)
                                            prefix_len_with_image = num_image_tokens + prefix_len
                                            nlm_labels[prefix_len_with_image:] = nlm_input_ids[0, prefix_len_with_image:]

                                            subtask_lm_inputs = {
                                                'input_ids': nlm_input_ids,
                                                'attention_mask': nlm_attention_mask,
                                                'labels': nlm_labels.unsqueeze(0),
                                                'pixel_values': vlm_inputs.get('pixel_values') if vlm_inputs is not None else None,
                                                'image_grid_thw': vlm_inputs.get('image_grid_thw') if vlm_inputs is not None else None,
                                            }
                                    except Exception as e2:
                                        logger.debug(f"Failed to build subtask data: {e2}")
                                        subtask_lm_inputs = None

                        # 进度目标: 归一化的子任务位置 [0, 1]
                        num_phases = len(checkpoints) + 1 if checkpoints else 1
                        progress_target = float(phase_idx) / max(num_phases - 1, 1)
                    except Exception as e:
                        logger.debug(f"Failed to load phase_info from {phase_path}: {e}")

                result = {
                    'first_frame': first_frame.squeeze(0),
                    'video_frames': video_frames,
                    'initial_state': initial_state,
                    'action_sequence': action_sequence,
                    'language_embedding': language_embedding,
                    'vlm_inputs': vlm_inputs,
                }

                if subtask_prompt is not None:
                    result['subtask_prompt'] = subtask_prompt
                if subtask_lm_inputs is not None:
                    result['subtask_lm_inputs'] = subtask_lm_inputs
                    # mot_decoder 格式字段
                    result['subtask_input_ids'] = subtask_dec_input_ids   # [1, max_subtask_len]
                    result['subtask_labels'] = subtask_dec_labels         # [1, max_subtask_len]
                if progress_target is not None:
                    result['progress_target'] = torch.tensor([progress_target], dtype=torch.float32)

                return result

            except Exception as e:
                logger.warning(f"Retry due to sample error ({episode_data.get('episode_name','?')}): {e}")
                continue

        return None

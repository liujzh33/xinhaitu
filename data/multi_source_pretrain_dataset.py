#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Source Pre-training Dataset for Motus
Unified loader for: RoboCoin, RoboTwin2.0, GM-100, RC1.0, RDT, RoboMIND, RoboMIND2.0, RC2.0
"""

import os
import random
import numpy as np
import torch
import torch.utils.data as data
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import warnings

from data.utils.image_utils import (
    tensor_to_pil, apply_image_augmentation,
    load_video_frames, get_video_frame_count
)
from utils.vlm_utils import preprocess_vlm_messages
from transformers import AutoProcessor

warnings.filterwarnings("ignore", category=FutureWarning, message=".*multichannel.*")

logger = logging.getLogger(__name__)


def unify_action_to_16(action: np.ndarray) -> np.ndarray:
    """
    统一 action 到 16 维: [left_joints(6) + pad(1) + left_grip(1) + right_joints(6) + pad(1) + right_grip(1)]

    | 原始 dim | 输入 layout              | 输出 16 维 layout              |
    |----------|-------------------------|--------------------------------|
    | 7        | [j0..j5, grip]          | [j0..j5, 0, grip, 0*8]         |
    | 8        | [j0..j6, grip]          | [j0..j6, grip, 0*8]            |
    | 14       | [j0..j5, g0, j7..j12, g1] | [j0..j5, 0, g0, j7..j12, 0, g1] |
    | 16       | 不变                    | 不变                           |
    """
    dim = action.shape[1]
    if dim == 16:
        return action
    elif dim == 14:
        # [j0..j5, g0, j6..j11, g1] -> [j0..j5, 0, g0, j6..j11, 0, g1]
        left_joints = action[:, :6]        # 6
        g0 = action[:, 6:7]                # 1
        right_joints = action[:, 7:13]     # 6 (j7..j12)
        g1 = action[:, 13:14]              # 1
        pad = np.zeros((action.shape[0], 1))
        return np.concatenate([left_joints, pad, g0, right_joints, pad, g1], axis=1)
    elif dim == 7:
        # [j0..j5, grip] -> [j0..j5, 0, grip, 0*8]
        left_joints = action[:, :6]        # 6
        grip = action[:, 6:7]              # 1
        pad1 = np.zeros((action.shape[0], 1))  # 1 pad after left joints
        right_pad = np.zeros((action.shape[0], 8))  # 8 zeros at end
        return np.concatenate([left_joints, pad1, grip, right_pad], axis=1)
    elif dim == 8:
        # [j0..j6, grip] -> [j0..j6, grip, 0*8]
        left_joints = action[:, :7]        # 7 joints
        grip = action[:, 7:8]              # 1 gripper
        right_pad = np.zeros((action.shape[0], 8))  # 8 zeros at end
        return np.concatenate([left_joints, grip, right_pad], axis=1)
    else:
        # 通用方式: 整体 pad 到 16
        return np.pad(action, ((0, 0), (0, 16 - dim)), mode='constant')


# =============================================================================
# Scanner functions for different data source patterns
# =============================================================================

def scan_robotwin2(source_dir: Path, robot_type: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Pattern: RoboTwin2.0, RC1.0
    Structure: {split}/{task}/{metas,qpos,umt5_wan,videos}/episode_id.pt
    Meta file: {split}/{task}/metas/episode_id.txt
    For RoboTwin2.0: each dataset_dir is already robot-specific (ur5, franka, etc.)
    For RC1.0: use scan_rc2 instead (has robot_type filtering)
    robot_type param for API consistency (prefixes task_name when set)
    """
    episodes = []
    for split_dir in source_dir.iterdir():
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name
        for task_dir in split_dir.iterdir():
            if not task_dir.is_dir():
                continue
            task_name = task_dir.name
            qpos_dir = task_dir / "qpos"
            videos_dir = task_dir / "videos"
            umt5_dir = task_dir / "umt5_wan"
            metas_dir = task_dir / "metas"

            if not (qpos_dir.exists() and videos_dir.exists() and umt5_dir.exists()):
                continue

            for qpos_file in qpos_dir.glob("*.pt"):
                ep_name = qpos_file.stem
                video_file = videos_dir / f"{ep_name}.mp4"
                lang_file = umt5_dir / f"{ep_name}.pt"
                meta_file = metas_dir / f"{ep_name}.txt" if metas_dir.exists() else None

                if video_file.exists() and lang_file.exists():
                    episodes.append({
                        'episode_name': ep_name,
                        'task_name': task_name,
                        'split': split_name,
                        'qpos_path': str(qpos_file),
                        'video_path': str(video_file),
                        'lang_path': str(lang_file),
                        'meta_path': str(meta_file) if meta_file and meta_file.exists() else None,
                    })
    return episodes


def scan_robocoin(source_dir: Path, robot_type: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Pattern: RoboCoin
    Structure: {task}/{episode_id}/{metas,qpos,umt5_wan,videos}/episode_id.*
    T5 file: trajectory.pt (not {episode_id}.pt)

    robot_type: if set, only collect episodes where task name starts with robot_type prefix
    e.g., robot_type='AgiBot-g1' -> only tasks like 'AgiBot-g1_battery_storage_b'
    """
    episodes = []
    for task_dir in source_dir.iterdir():
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        # Filter by robot_type if specified (e.g., only 'AgiBot-g1_*' tasks)
        if robot_type and not task_name.startswith(robot_type):
            continue
        for ep_dir in task_dir.iterdir():
            if not ep_dir.is_dir():
                continue
            ep_name = ep_dir.name
            qpos_dir = ep_dir / "qpos"
            videos_dir = ep_dir / "videos"
            umt5_dir = ep_dir / "umt5_wan"
            metas_dir = ep_dir / "metas"

            if not (qpos_dir.exists() and videos_dir.exists() and umt5_dir.exists()):
                continue

            # RoboCoin: files inside subdirs named same as episode, T5 is trajectory.pt
            qpos_file = qpos_dir / f"{ep_name}.pt"
            video_file = videos_dir / f"{ep_name}.mp4"
            lang_file = umt5_dir / "trajectory.pt"  # RoboCoin uses trajectory.pt
            meta_file = metas_dir / f"{ep_name}.txt" if metas_dir.exists() else None

            if qpos_file.exists() and video_file.exists() and lang_file.exists():
                episodes.append({
                    'episode_name': ep_name,
                    'task_name': task_name,
                    'split': 'default',
                    'robot_type': task_name.split('_')[0],  # e.g., 'Galaxea', 'AgiBot-g1', 'Cobot', etc.
                    'qpos_path': str(qpos_file),
                    'video_path': str(video_file),
                    'lang_path': str(lang_file),
                    'meta_path': str(meta_file) if meta_file and meta_file.exists() else None,
                })
    return episodes


def scan_gm100(source_dir: Path, robot_type: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Pattern: GM-100
    Structure: {task}/{episode_id}/{metas,action,umt5_wan,videos}/episode_id.*
    Note: uses 'action' subdir instead of 'qpos'
    Each dataset_dir is already robot-specific subdir (e.g., GM-100_AgiBotG1_converted)
    """
    episodes = []
    for task_dir in source_dir.iterdir():
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        for ep_dir in task_dir.iterdir():
            if not ep_dir.is_dir():
                continue
            ep_name = ep_dir.name
            # GM-100: some use 'action', some use 'qpos' subdir
            action_dir = ep_dir / "action"
            qpos_dir = ep_dir / "qpos"
            videos_dir = ep_dir / "videos"
            umt5_dir = ep_dir / "umt5_wan"
            metas_dir = ep_dir / "metas"

            # Determine which subdir exists for action/qpos data
            if action_dir.exists():
                state_dir = action_dir
                state_file = state_dir / f"{ep_name}.pt"
            elif qpos_dir.exists():
                state_dir = qpos_dir
                state_file = state_dir / f"{ep_name}.pt"
            else:
                continue

            if not (state_dir.exists() and videos_dir.exists() and umt5_dir.exists()):
                continue

            video_file = videos_dir / f"{ep_name}.mp4"
            lang_file = umt5_dir / "trajectory.pt"  # GM-100 uses trajectory.pt
            meta_file = metas_dir / f"{ep_name}.txt" if metas_dir.exists() else None

            if state_file.exists() and video_file.exists() and lang_file.exists():
                episodes.append({
                    'episode_name': ep_name,
                    'task_name': task_name,
                    'split': 'default',
                    'robot_type': robot_type,
                    'qpos_path': str(state_file),  # reuse qpos_path field for state (action/qpos)
                    'video_path': str(video_file),
                    'lang_path': str(lang_file),
                    'meta_path': str(meta_file) if meta_file and meta_file.exists() else None,
                })
    return episodes


def scan_rdt(source_dir: Path, robot_type: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Pattern: RDT
    Structure: {task}/{episode_id}/{metas,qpos,umt5_wan,videos}/episode_id.*
    Single robot type (robot_type='default'), robot_type param for API consistency only
    """
    episodes = []
    for task_dir in source_dir.iterdir():
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        for ep_dir in task_dir.iterdir():
            if not ep_dir.is_dir():
                continue
            ep_name = ep_dir.name
            qpos_dir = ep_dir / "qpos"
            videos_dir = ep_dir / "videos"
            umt5_dir = ep_dir / "umt5_wan"
            metas_dir = ep_dir / "metas"

            if not (qpos_dir.exists() and videos_dir.exists() and umt5_dir.exists()):
                continue

            qpos_file = qpos_dir / f"{ep_name}.pt"
            video_file = videos_dir / f"{ep_name}.mp4"
            lang_file = umt5_dir / "trajectory.pt"  # RDT uses trajectory.pt
            meta_file = metas_dir / f"{ep_name}.txt" if metas_dir.exists() else None

            if qpos_file.exists() and video_file.exists() and lang_file.exists():
                episodes.append({
                    'episode_name': ep_name,
                    'task_name': task_name,
                    'split': 'default',
                    'qpos_path': str(qpos_file),
                    'video_path': str(video_file),
                    'lang_path': str(lang_file),
                    'meta_path': str(meta_file) if meta_file and meta_file.exists() else None,
                })
    return episodes


def scan_robomind(source_dir: Path, robot_type: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Pattern: RoboMIND
    Structure: {robot_type}/{task}/{task}/success_episodes/{train,val}/{episode_id}/{metas,qpos,umt5_wan,videos}/

    robot_type: if set, only collect episodes from this robot type (e.g., 'agilex_3rgb')
    """
    episodes = []
    for robot_dir in source_dir.iterdir():
        if not robot_dir.is_dir():
            continue
        # Filter by robot_type if specified
        if robot_type and robot_dir.name != robot_type:
            continue
        for task_dir in robot_dir.iterdir():
            if not task_dir.is_dir():
                continue
            episodes_dir = task_dir / task_dir.name / "success_episodes"
            if not episodes_dir.exists():
                continue
            for split_dir in episodes_dir.iterdir():
                if not split_dir.is_dir():
                    continue
                split_name = split_dir.name
                for ep_dir in split_dir.iterdir():
                    if not ep_dir.is_dir():
                        continue
                    ep_name = ep_dir.name
                    qpos_dir = ep_dir / "qpos"
                    videos_dir = ep_dir / "videos"
                    umt5_dir = ep_dir / "umt5_wan"
                    metas_dir = ep_dir / "metas"

                    if not (qpos_dir.exists() and videos_dir.exists() and umt5_dir.exists()):
                        continue

                    qpos_file = qpos_dir / f"{ep_name}.pt"
                    video_file = videos_dir / f"{ep_name}.mp4"
                    lang_file = umt5_dir / "trajectory.pt"  # RoboMIND uses trajectory.pt
                    meta_file = metas_dir / f"{ep_name}.txt" if metas_dir.exists() else None

                    if qpos_file.exists() and video_file.exists() and lang_file.exists():
                        episodes.append({
                            'episode_name': ep_name,
                            'task_name': task_dir.name,
                            'split': split_name,
                            'robot_type': robot_dir.name,
                            'qpos_path': str(qpos_file),
                            'video_path': str(video_file),
                            'lang_path': str(lang_file),
                            'meta_path': str(meta_file) if meta_file and meta_file.exists() else None,
                        })
    return episodes


def scan_robomind2(source_dir: Path, robot_type: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Pattern: RoboMIND2.0
    Structure: {dataset}/{task}/{episode_id}/{metas,qpos,umt5_wan,videos}/
    Single robot type (Franka-sim), robot_type param for API consistency only
    """
    episodes = []
    for dataset_dir in source_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        for task_dir in dataset_dir.iterdir():
            if not task_dir.is_dir():
                continue
            task_name = task_dir.name
            for ep_dir in task_dir.iterdir():
                if not ep_dir.is_dir():
                    continue
                ep_name = ep_dir.name
                qpos_dir = ep_dir / "qpos"
                videos_dir = ep_dir / "videos"
                umt5_dir = ep_dir / "umt5_wan"
                metas_dir = ep_dir / "metas"

                if not (qpos_dir.exists() and videos_dir.exists() and umt5_dir.exists()):
                    continue

                qpos_file = qpos_dir / f"{ep_name}.pt"
                video_file = videos_dir / f"{ep_name}.mp4"
                lang_file = umt5_dir / "trajectory.pt"  # RoboMIND2.0 uses trajectory.pt
                meta_file = metas_dir / f"{ep_name}.txt" if metas_dir.exists() else None

                if qpos_file.exists() and video_file.exists() and lang_file.exists():
                    episodes.append({
                        'episode_name': ep_name,
                        'task_name': task_name,
                        'split': 'default',
                        'qpos_path': str(qpos_file),
                        'video_path': str(video_file),
                        'lang_path': str(lang_file),
                        'meta_path': str(meta_file) if meta_file and meta_file.exists() else None,
                    })
    return episodes


def scan_rc2(source_dir: Path, robot_type: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Pattern: RC2.0 (RoboChallenge 2.0)
    Structure: {robot_type}/{task}/{episode_id}/{metas,qpos,umt5_wan,videos}/
    e.g., RoboChallenge2.0_ALOHA/pack_the_items/episode_000102/
    NOTE: T5 embedding is trajectory.pt (not {episode_id}.pt)
    """
    episodes = []
    for robot_dir in source_dir.iterdir():
        if not robot_dir.is_dir():
            continue
        robot_name = robot_dir.name  # e.g., RoboChallenge2.0_ALOHA
        if robot_type and robot_type not in robot_name:
            continue
        for task_dir in robot_dir.iterdir():
            if not task_dir.is_dir():
                continue
            task_name = task_dir.name
            for ep_dir in task_dir.iterdir():
                if not ep_dir.is_dir():
                    continue
                ep_name = ep_dir.name  # e.g., episode_000102
                qpos_dir = ep_dir / "qpos"
                videos_dir = ep_dir / "videos"
                umt5_dir = ep_dir / "umt5_wan"
                metas_dir = ep_dir / "metas"

                if not (qpos_dir.exists() and videos_dir.exists() and umt5_dir.exists()):
                    continue

                qpos_file = qpos_dir / f"{ep_name}.pt"
                video_file = videos_dir / f"{ep_name}.mp4"
                lang_file = umt5_dir / "trajectory.pt"  # RC2.0 uses trajectory.pt
                meta_file = metas_dir / f"{ep_name}.txt" if metas_dir.exists() else None

                if qpos_file.exists() and video_file.exists() and lang_file.exists():
                    episodes.append({
                        'episode_name': ep_name,
                        'task_name': f"{robot_name}_{task_name}",
                        'split': 'default',
                        'qpos_path': str(qpos_file),
                        'video_path': str(video_file),
                        'lang_path': str(lang_file),
                        'meta_path': str(meta_file) if meta_file and meta_file.exists() else None,
                    })
    return episodes


def scan_dobot(source_dir: Path, robot_type: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Pattern: Dobot (LeRobot v2.1 converted)
    Structure: {dataset_dir}/{task_name}/{episode_id}/{metas,qpos,umt5_wan,videos}/
    e.g., dobot_pour_water_full/dobot_pour_water_full/episode_000000/
    Files: metas/episode_XXX.txt, qpos/episode_XXX.pt, umt5_wan/trajectory.pt, videos/episode_XXX.mp4
    """
    episodes = []
    for task_dir in source_dir.iterdir():
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        if robot_type and robot_type not in task_name:
            continue
        for ep_dir in task_dir.iterdir():
            if not ep_dir.is_dir():
                continue
            ep_name = ep_dir.name
            if not ep_name.startswith('episode_'):
                continue

            metas_dir = ep_dir / 'metas'
            qpos_dir = ep_dir / 'qpos'
            umt5_dir = ep_dir / 'umt5_wan'
            videos_dir = ep_dir / 'videos'

            meta_file = metas_dir / f"{ep_name}.txt" if metas_dir.exists() else None
            qpos_file = qpos_dir / f"{ep_name}.pt" if qpos_dir.exists() else None
            lang_file = umt5_dir / "trajectory.pt" if umt5_dir.exists() else None
            video_file = videos_dir / f"{ep_name}.mp4" if videos_dir.exists() else None

            if qpos_file.exists() and video_file.exists() and lang_file.exists():
                episodes.append({
                    'episode_name': ep_name,
                    'task_name': task_name,
                    'split': 'default',
                    'qpos_path': str(qpos_file),
                    'video_path': str(video_file),
                    'lang_path': str(lang_file),
                    'meta_path': str(meta_file) if meta_file and meta_file.exists() else None,
                })
    return episodes


# Scanner registry
SCANNERS = {
    'robotwin2': scan_robotwin2,
    'robocoin': scan_robocoin,
    'gm100': scan_gm100,
    'rdt': scan_rdt,
    'robomind': scan_robomind,
    'robomind2': scan_robomind2,
    'rc2': scan_rc2,
    'dobot': scan_dobot,
}


# =============================================================================
# Dataset Class
# =============================================================================

class MultiSourcePretrainDataset(data.Dataset):
    """
    Unified multi-source pre-training dataset.

    Supported sources:
    - RoboCoin: {task}/{episode}/{metas,qpos,umt5_wan,videos}/
    - RoboTwin2.0, RC1.0: {split}/{task}/{metas,qpos,umt5_wan,videos}/episode_id.pt
    - GM-100: {task}/{episode}/{metas,action,umt5_wan,videos}/
    - RDT: {task}/{episode}/{metas,qpos,umt5_wan,videos}/
    - RoboMIND: {robot}/{task}/{task}/success_episodes/{train,val}/{episode}/
    - RoboMIND2.0: {dataset}/{task}/{episode}/

    Output format (compatible with existing datasets):
    {
        'first_frame':       # [C, H, W] tensor
        'video_frames':      # [F, C, H, W] tensor
        'action_sequence':   # [F*ratio, 16] tensor (unified to 16-dim)
        'language_embedding':# [seq_len, 4096] tensor
        'initial_state':     # [16] tensor
    }
    """

    def __init__(
        self,
        sources: List[Dict[str, Any]],
        global_downsample_rate: int = 3,
        video_action_freq_ratio: int = 5,
        num_video_frames: int = 3,
        video_size: Tuple[int, int] = (384, 320),
        max_episodes: Optional[int] = None,
        val: bool = False,
        image_aug: bool = False,
        max_action_dim: int = 16,
        vlm_checkpoint_path: Optional[str] = None,
    ):
        """
        Args:
            sources: List of source configs, each containing:
                - dataset_dir: str
                - pattern: str ('robotwin2', 'robocoin', 'gm100', 'rdt', 'robomind', 'robomind2')
                - weight: float (sampling weight)
            global_downsample_rate: e.g. 3 for 30Hz->10Hz
            video_action_freq_ratio: Video:Action frequency ratio
            num_video_frames: Number of video frames to predict
            video_size: (H, W)
            max_episodes: Max total episodes to load (for debugging)
            val: Validation mode (no augmentation)
            image_aug: Apply image augmentation
            max_action_dim: Target action dimension (default 16)
        """
        self.global_downsample_rate = global_downsample_rate
        self.video_action_freq_ratio = video_action_freq_ratio
        self.num_video_frames = num_video_frames
        self.video_size = video_size
        self.val = val
        self.image_aug = image_aug and not val
        self.max_action_dim = max_action_dim

        self.action_chunk_size = num_video_frames * video_action_freq_ratio

        # Initialize VLM processor for complete VLM processing in dataset
        self.vlm_processor = None
        if vlm_checkpoint_path is not None:
            try:
                self.vlm_processor = AutoProcessor.from_pretrained(vlm_checkpoint_path)
                logger.info(f"VLM processor loaded from {vlm_checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to load VLM processor from {vlm_checkpoint_path}: {e}")
                logger.warning("VLM processing will be disabled for this dataset instance")
        else:
            logger.info("VLM checkpoint path not provided, VLM processing disabled")

        # Collect all episodes from all sources
        self.episodes = []  # List of (episode_data, source_weight)
        total_found = 0

        for source in sources:
            dataset_dir = Path(source['dataset_dir'])
            pattern = source.get('pattern', 'robotwin2')
            weight = source.get('weight', 1.0)
            robot_type = source.get('robot_type', None)

            if not dataset_dir.exists():
                logger.warning(f"Source dir not found: {dataset_dir}")
                continue

            scanner = SCANNERS.get(pattern)
            if scanner is None:
                logger.warning(f"Unknown pattern '{pattern}' for {dataset_dir}, skipping")
                continue

            eps = scanner(dataset_dir, robot_type=robot_type)
            logger.info(f"[{pattern}] {dataset_dir.name} (robot_type={robot_type}): found {len(eps)} episodes, weight={weight}")
            for ep in eps:
                self.episodes.append((ep, weight))
            total_found += len(eps)

        if max_episodes is not None and total_found > max_episodes:
            # Weighted sampling to limit
            weights = [w for _, w in self.episodes]
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            indices = np.random.choice(len(self.episodes), max_episodes, replace=False, p=probs)
            self.episodes = [self.episodes[i] for i in indices]
            logger.info(f"Limited to {max_episodes} episodes (from {total_found})")

        self.total_episodes = len(self.episodes)
        logger.info(f"Total episodes: {self.total_episodes}")

    def _calculate_sampling_indices(self, total_frames: int) -> Tuple[int, List[int], List[int]]:
        """Calculate condition frame, video indices, action indices."""
        physical_chunk_size = self.action_chunk_size * self.global_downsample_rate
        max_condition_idx = total_frames - physical_chunk_size - 1

        if max_condition_idx < 0:
            condition_frame_idx = 0
        else:
            condition_frame_idx = random.randint(0, max_condition_idx)

        # Action indices
        action_indices = []
        for i in range(self.action_chunk_size):
            action_idx = condition_frame_idx + (i + 1) * self.global_downsample_rate
            action_indices.append(min(action_idx, total_frames - 1))

        # Video indices
        video_indices = []
        for i in range(self.num_video_frames):
            action_step = (i + 1) * self.video_action_freq_ratio - 1
            if action_step < len(action_indices):
                video_indices.append(action_indices[action_step])
            else:
                video_indices.append(action_indices[-1])

        return condition_frame_idx, video_indices, action_indices

    def _load_action(self, qpos_path: str, action_indices: List[int], initial_state_idx: int, robot_type: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and unify action data to 16-dim."""
        qpos_data = torch.load(qpos_path, map_location='cpu').numpy()  # [T, D]

        # Unify to 16-dim
        qpos_data = unify_action_to_16(qpos_data)

        # Binarize gripper for RoboMIND/agilex_3rgb: >2.5 -> 1, <2.5 -> 0
        # After unify_action_to_16, gripper positions are at index 7 (left) and 15 (right)
        if robot_type == 'agilex_3rgb':
            gripper_indices = [7, 15]  # left_grip, right_grip in 16-dim layout
            for idx in gripper_indices:
                qpos_data[:, idx] = (qpos_data[:, idx] > 2.5).astype(np.float32)

        # Scale gripper for GM-100 variants (gripper at index 7 and 15 in 16-dim layout)
        if robot_type == 'AgileX':
            # dim6/13 [0, 0.08] -> *10 -> [0, 0.8]
            for idx in [7, 15]:
                qpos_data[:, idx] = qpos_data[:, idx] * 10.0
        elif robot_type == 'GalaxeaR1Pro':
            # dim7/15 [0, 80] -> /100 -> [0, 0.8]
            for idx in [7, 15]:
                qpos_data[:, idx] = qpos_data[:, idx] / 100.0

        # Normalize gripper for RoboCoin/Galaxea (dual-arm, gripper at index 7 and 15 after 14->16 dim unification)
        # Original 14-dim: [j0..j5, g0, j7..j12, g1] -> g0 at index 6, g1 at index 13
        # After unify: g0 -> index 7, g1 -> index 15
        # Range [0, 100] with some negative values, normalize by /100
        if robot_type == 'Galaxea' or robot_type == 'R1':
            for idx in [7, 15]:
                qpos_data[:, idx] = qpos_data[:, idx] / 100.0

        if initial_state_idx >= len(qpos_data):
            initial_state_idx = len(qpos_data) - 1
        initial_state = torch.from_numpy(qpos_data[initial_state_idx]).float()

        actions = []
        for idx in action_indices:
            if idx >= len(qpos_data):
                actions.append(qpos_data[-1])
            else:
                actions.append(qpos_data[idx])
        action_sequence = torch.from_numpy(np.stack(actions)).float()
        return initial_state, action_sequence

    def _load_language_embedding(self, lang_path: str) -> torch.Tensor:
        """Load pre-encoded T5 language embedding."""
        try:
            emb_data = torch.load(lang_path, map_location='cpu')
            # emb_data may be a list of tensors or a single tensor
            if isinstance(emb_data, list):
                emb = emb_data[0]  # Take first
            else:
                emb = emb_data
            if emb.dim() == 3:
                emb = emb.squeeze(0)
            return emb
        except Exception as e:
            logger.error(f"Error loading T5 embedding from {lang_path}: {e}")
            # Return zeros as fallback
            return torch.zeros(512, 4096)

    def _load_meta_text(self, meta_path: str) -> str:
        """Load text instruction from meta file (first non-empty line)."""
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            return lines[0] if lines else ""
        except Exception as e:
            logger.warning(f"Error loading meta text from {meta_path}: {e}")
            return ""

    def _load_vlm_inputs(self, text_instruction: str, first_frame: torch.Tensor) -> Optional[Dict[str, Any]]:
        """Generate VLM inputs from text instruction and first frame."""
        if self.vlm_processor is None or not text_instruction:
            return None
        try:
            first_frame_pil = tensor_to_pil(first_frame)
            return preprocess_vlm_messages(text_instruction, first_frame_pil, self.vlm_processor)
        except Exception as e:
            logger.warning(f"Error generating VLM inputs: {e}")
            return None

    def __len__(self) -> int:
        return self.total_episodes * 10

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get a training sample."""
        max_attempts = 8
        for _ in range(max_attempts):
            if not self.episodes:
                return None
            episode_data, weight = random.choice(self.episodes)

            try:
                total_frames = get_video_frame_count(episode_data['video_path'])
                if total_frames < 2:
                    continue

                condition_frame_idx, video_indices, action_indices = self._calculate_sampling_indices(total_frames)

                first_frame = load_video_frames(episode_data['video_path'], [condition_frame_idx], self.video_size)
                video_frames = load_video_frames(episode_data['video_path'], video_indices, self.video_size)
                initial_state, action_sequence = self._load_action(episode_data['qpos_path'], action_indices, condition_frame_idx, episode_data.get('robot_type'))
                language_embedding = self._load_language_embedding(episode_data['lang_path'])

                # VLM processing: load meta text and generate vlm_inputs
                meta_text = ""
                vlm_inputs = None
                if episode_data.get('meta_path'):
                    meta_text = self._load_meta_text(episode_data['meta_path'])
                if meta_text and self.vlm_processor is not None:
                    vlm_inputs = self._load_vlm_inputs(meta_text, first_frame.squeeze(0))

                return {
                    'first_frame': first_frame.squeeze(0),
                    'video_frames': video_frames,
                    'initial_state': initial_state,
                    'action_sequence': action_sequence,
                    'language_embedding': language_embedding,
                    'vlm_inputs': vlm_inputs,
                }

            except Exception as e:
                logger.warning(f"Retry ({episode_data.get('episode_name', '?')}): {e}")
                continue

        return None

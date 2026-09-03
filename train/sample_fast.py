#!/usr/bin/env python3
"""
Evaluation utilities for Motus (FastWAM-style action-only inference).
Implements action prediction metrics computation for validation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from collections import defaultdict
import logging
import time

logger = logging.getLogger(__name__)


# Note: create_video_grid and video visualization removed since we only predict actions


@torch.no_grad()
def inference_sample(model, batch: Dict, config) -> Tuple[torch.Tensor, float]:
    """
    Run inference to predict actions only (FastWAM-style, no video generation).

    Args:
        model: MotusWanVlmDirectMask model
        batch: Input batch containing observations, states, language embeddings, text instructions
        config: Configuration object containing inference parameters

    Returns:
        predicted_actions: (B, action_chunk_size, action_dim)
        inference_time: Time taken for inference in seconds
    """
    model.eval()

    # Extract inference parameters from config
    num_inference_steps = config.model.inference.num_inference_timesteps
    action_horizon = batch.get('action_sequence', torch.empty(1, 1, 14)).shape[1]

    # Move batch data to device
    device = next(model.parameters()).device
    first_frame = batch['first_frame'].to(device)  # [B, C, H, W] - conditioning frame

    state = batch['initial_state'].to(device) if 'initial_state' in batch and batch['initial_state'] is not None else None

    language_embeddings = batch['language_embedding']
    if language_embeddings is not None:
        language_embeddings = language_embeddings.to(device)

    vlm_inputs = batch['vlm_inputs']
    if vlm_inputs is not None:
        # Move all tensors in the VLM inputs dict to device
        vlm_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in vlm_inputs.items()}

    # Warmup run to avoid timing overhead from lazy initialization
    with torch.no_grad():
        _ = model.infer_action_only(
            first_frame=first_frame,
            state=state,
            action_horizon=action_horizon,
            num_inference_steps=num_inference_steps,
            language_embeddings=language_embeddings,
            vlm_inputs=vlm_inputs,
        )

    # Timing run
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        predicted_actions = model.infer_action_only(
            first_frame=first_frame,
            state=state,
            action_horizon=action_horizon,
            num_inference_steps=num_inference_steps,
            language_embeddings=language_embeddings,
            vlm_inputs=vlm_inputs,
        )

    torch.cuda.synchronize()
    inference_time = time.time() - start_time

    model.train()
    return predicted_actions, inference_time


def compute_action_metrics(predicted_actions: torch.Tensor, ground_truth_actions: torch.Tensor) -> Dict[str, float]:
    """
    Compute action prediction metrics (MSE and L2 error).
    
    Args:
        predicted_actions: (B, T, action_dim) predicted actions
        ground_truth_actions: (B, T, action_dim) ground truth actions  
        
    Returns:
        Dictionary containing MSE and L2 error metrics
    """
    # Compute MSE loss
    mse_loss = F.mse_loss(predicted_actions, ground_truth_actions, reduction='none').float()
    mse_loss_per_sample = mse_loss.reshape(predicted_actions.shape[0], -1).mean(1)
    
    # Compute L2 error (RMSE)
    l2_loss = mse_loss.sqrt() / (1 + 1e-3)
    l2_loss_per_sample = l2_loss.reshape(predicted_actions.shape[0], -1).mean(1)
    
    return {
        'mse_loss': mse_loss_per_sample.mean().item(),
        'l2_error': l2_loss_per_sample.mean().item(),
        'mse_std': mse_loss_per_sample.std().item(),
        'l2_std': l2_loss_per_sample.std().item()
    }


@torch.no_grad()
def evaluate_model(model, dataloader, accelerator, config, num_eval_batches: int = 2) -> Dict[str, float]:
    """
    Local-only evaluation: no distributed aggregation; safe for rank0-only evaluation.
    FastWAM-style: Only evaluate action prediction, no video generation.

    Returns:
        Dict containing action metrics and inference timing stats
    """
    logger.info(f"Running Action-only evaluation for {num_eval_batches} batches...")
    model.eval()

    from collections import defaultdict
    metrics = defaultdict(list)

    for step, batch in enumerate(dataloader):
        if step >= num_eval_batches:
            break
        if batch is None:
            continue

        # Inference (action only, no video generation)
        predicted_actions, inference_time = inference_sample(model, batch, config)

        # Track inference time
        batch_size = predicted_actions.shape[0]
        metrics['inference_time_per_batch'].append(inference_time)
        metrics['inference_time_per_sample'].append(inference_time / batch_size)

        # Action metrics only
        if 'action_sequence' in batch and predicted_actions is not None:
            gt_actions = batch['action_sequence'][:, :predicted_actions.shape[1]].to(predicted_actions.device)
            action_metrics = compute_action_metrics(predicted_actions, gt_actions)
            for key, value in action_metrics.items():
                metrics[f'action_{key}'].append(value)

    # Aggregate metrics
    final_metrics = {}
    for key, values in metrics.items():
        if values:
            final_metrics[key] = float(np.mean(values))
            # Only compute std for action metrics, not for inference time
            if key.startswith('action_'):
                final_metrics[f'{key}_std'] = float(np.std(values))

    model.train()
    return final_metrics


def log_evaluation_metrics(metrics: Dict, writer, accelerator, global_step: int):
    """
    Log evaluation metrics to tensorboard and wandb.

    Args:
        metrics: Dictionary containing evaluation metrics
        writer: TensorBoard writer (can be None)
        accelerator: HuggingFace accelerator
        global_step: Current training step
    """
    if accelerator.is_main_process:
        # Log scalar metrics (action only)
        log_dict = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                log_dict[f'eval/{key}'] = value

        # Log to accelerator (wandb)
        if log_dict:
            accelerator.log(log_dict, step=global_step)

        # Log to TensorBoard
        if writer is not None:
            for key, value in log_dict.items():
                writer.add_scalar(key, value, global_step)

        # Print summary
        logger.info("=== Action Evaluation Results (FastWAM-style) ===")
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and '_std' not in key:
                logger.info(f"  {key}: {value:.4f}")
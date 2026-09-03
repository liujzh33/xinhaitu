# Fast Action Tokenizer Adapter
"""
Adapted from https://huggingface.co/physical-intelligence/fast

This module converts continuous/discrete robot actions to/from
pseudo-natural language token strings like <robot_action_12><robot_action_3>...
This facilitates direct integration into multimodal large models (VLM/LLM).
"""

import torch.nn as nn
from typing import List, Optional, Tuple
import numpy as np
import sys
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FastActionTokenizer(nn.Module):
    """Fast Action Tokenizer for converting actions to discrete tokens."""

    def __init__(self, fast_tokenizer_name="physical-intelligence/fast"):
        super().__init__()

        if '/' in fast_tokenizer_name or '\\' in fast_tokenizer_name:
            # Local path
            self._load_from_local(fast_tokenizer_name)
        else:
            # HF Hub path
            self._load_from_hub(fast_tokenizer_name)

        # Store reference to bpe_tokenizer for direct access
        self.bpe_tokenizer = self.fast_tokenizer.bpe_tokenizer

        # Get action token range (from processor config)
        self.min_token = getattr(self.fast_tokenizer, 'min_token', -354)
        self.vocab_size = getattr(self.fast_tokenizer, 'vocab_size', 2048)

        logger.info(f"Fast tokenizer initialized: min_token={self.min_token}, vocab_size={self.vocab_size}")

    def _load_from_local(self, local_path: str):
        """Load fast tokenizer from local directory."""
        tokenizer_dir = Path(local_path).resolve()

        # Add to sys.path for importing processing_action_tokenizer
        if str(tokenizer_dir) not in sys.path:
            sys.path.insert(0, str(tokenizer_dir))

        # Load processor config
        processor_config_path = tokenizer_dir / "processor_config.json"
        if processor_config_path.exists():
            with open(processor_config_path) as f:
                processor_config = json.load(f)
            scale = processor_config.get('scale', 10.0)
            vocab_size = processor_config.get('vocab_size', 2048)
            min_token = processor_config.get('min_token', 0)
        else:
            scale = 10.0
            vocab_size = 2048
            min_token = 0

        # Import and create UniversalActionProcessor
        from processing_action_tokenizer import UniversalActionProcessor
        from transformers import PreTrainedTokenizerFast

        # Load BPE tokenizer
        bpe_tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))

        # Create processor
        self.fast_tokenizer = UniversalActionProcessor(
            bpe_tokenizer=bpe_tokenizer,
            scale=scale,
            vocab_size=vocab_size,
            min_token=min_token
        )
        logger.info(f"Loaded fast tokenizer from {local_path}")

    def _load_from_hub(self, hub_path: str):
        """Load fast tokenizer from HuggingFace Hub."""
        try:
            from transformers import AutoProcessor
            self.fast_tokenizer = AutoProcessor.from_pretrained(hub_path, trust_remote_code=True)
            self.bpe_tokenizer = self.fast_tokenizer.bpe_tokenizer
            self.min_token = getattr(self.fast_tokenizer, 'min_token', -354)
            self.vocab_size = getattr(self.fast_tokenizer, 'vocab_size', 2048)
            logger.info(f"Loaded fast tokenizer from hub: {hub_path}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer from hub {hub_path}: {e}")
            raise

    def encoder_action2fastoken(self, raw_actions: List[np.ndarray]) -> List[List[int]]:
        """
        Convert continuous actions to discrete Fast tokens (BPE token IDs).

        This is the original fast tokenizer encoder method.

        Args:
            raw_actions: List of action arrays, each shaped [T, D]

        Returns:
            List[List[int]]: BPE token IDs for each action sequence
        """
        if len(raw_actions) == 0:
            return []

        # Stack actions: (B, T, D)
        batch_actions = np.stack(raw_actions, axis=0)

        # Convert to float32 if needed
        if batch_actions.dtype not in [np.float32, np.float64]:
            batch_actions = batch_actions.astype(np.float32)

        # Use the fast tokenizer to discretize
        # Returns BPE token IDs: List[List[int]]
        batch_bpe_tokens = self.fast_tokenizer(batch_actions)

        return batch_bpe_tokens

    def decoder_action(self, bpe_tokens: List[List[int]], time_horizon: int, action_dim: int) -> np.ndarray:
        """
        Convert BPE token IDs back to continuous actions.

        Args:
            bpe_tokens: BPE token IDs from VLM
            time_horizon: Number of timesteps
            action_dim: Action dimension

        Returns:
            np.ndarray: Continuous actions [B, T, D]
        """
        return self.fast_tokenizer.decode(bpe_tokens, time_horizon=time_horizon, action_dim=action_dim)

    def map_bpe_tokens_to_vlm_action(self, bpe_token_ids: List[int]) -> str:
        """
        Map BPE token IDs to VLM action token format.

        We map each BPE token ID to a corresponding <robot_action_X> token.
        Since BPE tokens can be numerous, we use modulo to map to 0-2047 range.

        Args:
            bpe_token_ids: List of BPE token IDs

        Returns:
            str: VLM action format (e.g., "<robot_action_123><robot_action_456>...")
        """
        # Map BPE token IDs to action token range [0, 2047]
        if self.min_token < 0:
            # Apply min_token offset
            action_token_ids = [(t + abs(self.min_token)) % 2048 for t in bpe_token_ids]
        else:
            action_token_ids = [t % 2048 for t in bpe_token_ids]

        return ''.join([f"<robot_action_{tid}>" for tid in action_token_ids])

    def map_vlm_tokens_to_bpe(self, vlm_token_ids: List[int], action_token_min: int, action_token_max: int) -> List[int]:
        """
        Map VLM token IDs back to BPE token IDs.

        Args:
            vlm_token_ids: VLM token IDs (from generation)
            action_token_min: VLM action token range start
            action_token_max: VLM action token range end

        Returns:
            List[int]: BPE token IDs
        """
        bpe_token_ids = []
        for tid in vlm_token_ids:
            if action_token_min <= tid <= action_token_max:
                # Convert from VLM action token ID back to BPE token ID
                action_token_id = tid - action_token_min
                if self.min_token < 0:
                    bpe_token_id = action_token_id + self.min_token
                else:
                    bpe_token_id = action_token_id
                bpe_token_ids.append(bpe_token_id)

        return bpe_token_ids


def get_fast_tokenizer(config=None):
    """
    Factory function to create FastActionTokenizer.

    Args:
        config: Optional config with tokenizer settings

    Returns:
        FastActionTokenizer: Initialized tokenizer
    """
    fast_tokenizer_name = "physical-intelligence/fast"

    if config is not None:
        fast_tokenizer_name = getattr(config, 'fast_tokenizer_path', fast_tokenizer_name)

    return FastActionTokenizer(
        fast_tokenizer_name=fast_tokenizer_name
    )
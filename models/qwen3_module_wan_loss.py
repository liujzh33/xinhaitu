# Qwen3-VL Module for Direct MoT Participation with WAN
# Each layer has its own QKV projection to the unified WAN head space
# VLM has 28 layers, WAN has 30 layers - we map VLM Layer 26, 27 to WAN Layer 28, 29

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import logging

import sys
from pathlib import Path

BAK_ROOT = str((Path(__file__).parent.parent / "bak").resolve())
if str(BAK_ROOT) not in sys.path:
    sys.path.insert(0, str(BAK_ROOT))

from wan.modules.model import WanLayerNorm, WanRMSNorm

logger = logging.getLogger(__name__)


@dataclass
class Qwen3VLWanConfig:
    """Configuration for Qwen3-VL Expert model with WAN backbone."""
    vlm_dim: int = 2048              # Qwen3-VL-2B hidden size
    head_dim: int = 128              # Head dimension (for WAN 5B)
    num_heads: int = 24              # Number of heads (for WAN 5B)
    num_layers: int = 30             # Number of layers (matches WAN)
    eps: float = 1e-5                # Layer norm epsilon

    # Action token settings for VLM action loss
    action_token_min: int = -1       # Will be determined dynamically from tokenizer
    action_token_max: int = -1       # Will be determined dynamically from tokenizer
    enable_vlm_action_loss: bool = True  # Enable VLM action loss if tokens are available


class Qwen3VLWanBlock(nn.Module):
    """
    Qwen3-VL Expert Block - per-layer QKV projection for direct MoT participation.

    Unlike Understanding Expert which has its own FFN and is trained,
    this block only provides QKV projections to the unified WAN head space.
    Qwen3-VL itself can be trained (not frozen).

    Key differences from UndExpertBlock:
    - No FFN (Qwen3-VL has its own internal FFN)
    - No vlm_adapter (direct 2048D input, already matches VLM dim)
    - VLM can be trained (not frozen)
    - Projects to WAN head space (3072D = 24 × 128)
    """

    def __init__(self, vlm_dim: int, head_dim: int, num_heads: int, eps: float = 1e-5):
        super().__init__()
        self.vlm_dim = vlm_dim  # 2048 for Qwen3-VL-2B
        self.num_heads = num_heads  # 24 for WAN 5B
        self.head_dim = head_dim  # 128 for WAN 5B
        self.unified_dim = num_heads * head_dim  # 3072

        # LayerNorm (same style as WAN)
        self.norm1 = WanLayerNorm(vlm_dim, eps=eps)

        # QKV projection: 2048D -> Unified WAN head space (24 heads × 128 = 3072)
        self.wan_vlm_qkv = nn.Parameter(
            torch.randn(3, num_heads, vlm_dim, head_dim)
            / (vlm_dim * head_dim) ** 0.5
        )

        # Output projection: Unified dim (3072) -> VLM dim (2048)
        self.wan_vlm_o = nn.Linear(self.unified_dim, vlm_dim, bias=False)

        # Q/K normalization (RMSNorm like WAN)
        self.wan_vlm_norm_q = WanRMSNorm(self.unified_dim, eps=eps)
        self.wan_vlm_norm_k = WanRMSNorm(self.unified_dim, eps=eps)


class Qwen3VLWanModule(nn.Module):
    """
    Qwen3-VL Module - extracts per-layer hidden states for direct MoT with WAN.

    This module:
    1. Extracts all layer outputs from Qwen3-VL (28 layers)
    2. Maps 28 VLM layers to 30 WAN layers (Layer 26, 27 reused for WAN Layer 28, 29)
    3. Provides per-layer QKV projections to unified WAN head space
    4. Processes residual connections after MoT attention
    """

    def __init__(self, vlm_model, config: Qwen3VLWanConfig, dtype, device, tokenizer=None):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.device = device

        # VLM model reference
        self.vlm_model = vlm_model
        self.tokenizer = tokenizer  # External tokenizer (from main model)

        # Check if action tokens are available in tokenizer
        logger.info("=== Starting VLM action token detection ===")
        self._detect_action_tokens()
        logger.info(f"=== Detection complete: enable_vlm_action_loss={self.config.enable_vlm_action_loss}, "
                   f"action_token_min={self.config.action_token_min}, action_token_max={self.config.action_token_max} ===")

        # Create per-layer projection blocks (30 blocks for WAN)
        self.blocks = nn.ModuleList([
            Qwen3VLWanBlock(
                vlm_dim=config.vlm_dim,
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                eps=config.eps
            )
            for _ in range(config.num_layers)
        ])

        self.blocks.to(device=device, dtype=dtype)
        logger.info(f"Qwen3VLWanModule initialized with {config.num_layers} layers (WAN)")
        logger.info(f"  VLM dim: {config.vlm_dim}, WAN head space: {config.head_dim * config.num_heads}")
        logger.info(f"  Layer mapping: VLM 0-27 -> WAN 0-27, VLM 26->WAN 28, VLM 27->WAN 29")
        if self.config.enable_vlm_action_loss and self.config.action_token_min >= 0:
            logger.info(f"  VLM Action Loss: ENABLED (token range: [{self.config.action_token_min}, {self.config.action_token_max}])")
        else:
            logger.info(f"  VLM Action Loss: DISABLED (action tokens not found in tokenizer)")

    def _detect_action_tokens(self):
        """
        Detect if action tokens (<robot_action_0>...<robot_action_2047>) are in VLM tokenizer.
        Sets action_token_min and action_token_max in config.
        """
        if not self.config.enable_vlm_action_loss:
            return

        # Use external tokenizer if available
        tokenizer = self.tokenizer
        if tokenizer is None:
            logger.warning("No tokenizer provided - VLM action loss disabled")
            self.config.action_token_min = -1
            self.config.action_token_max = -1
            self.config.enable_vlm_action_loss = False
            return

        try:
            # Check if action tokens exist
            test_tokens = [f"<robot_action_{i}>" for i in [0, 2047]]
            unk_token_id = tokenizer.unk_token_id if hasattr(tokenizer, 'unk_token_id') else None

            token_ids = [tokenizer.convert_tokens_to_ids(t) for t in test_tokens]

            logger.info(f"[VLM Action Token Detection]")
            logger.info(f"  <robot_action_0> -> {token_ids[0]}")
            logger.info(f"  <robot_action_2047> -> {token_ids[1]}")
            logger.info(f"  UNK token ID: {unk_token_id}")

            # Check if all tokens are valid (not unknown)
            if unk_token_id is not None:
                all_valid = all(tid != unk_token_id for tid in token_ids)
            else:
                # No unk_token_id, check if tokens are valid (in vocab range)
                all_valid = all(tid >= 0 and tid < len(tokenizer) for tid in token_ids)

            if all_valid:
                self.config.action_token_min = token_ids[0]
                self.config.action_token_max = token_ids[1]
                logger.info(f"  ✓ Action tokens found: [{self.config.action_token_min}, {self.config.action_token_max}]")
            else:
                logger.warning(f"  ✗ Action tokens NOT found in tokenizer")
                logger.warning(f"  Token IDs: {token_ids}, UNK: {unk_token_id}")
                self.config.action_token_min = -1
                self.config.action_token_max = -1
                self.config.enable_vlm_action_loss = False
        except Exception as e:
            import traceback
            logger.warning(f"Failed to detect action tokens: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            self.config.action_token_min = -1
            self.config.action_token_max = -1
            self.config.enable_vlm_action_loss = False

    def extract_per_layer_features(
        self,
        vlm_inputs
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """
        Extract Qwen3-VL hidden states from ALL layers for per-layer MoT.

        VLM has 28 layers, but WAN has 30 layers.
        We map: VLM Layer 26 -> WAN Layer 28, VLM Layer 27 -> WAN Layer 29

        Args:
            vlm_inputs: VLM inputs (list of dicts or dict of tensors)

        Returns:
            Tuple of (hidden_states_list, vlm_loss):
                - hidden_states_list: 30 layers of hidden states [layer_0, ..., layer_27, layer_26, layer_27]
                  Each tensor: [B, seq_len, 2048]
                - vlm_loss: VLM action loss (None if no labels or not enabled)
        """
        import time
        timing = {}

        if isinstance(vlm_inputs, list):
            B = len(vlm_inputs)
        else:
            B = vlm_inputs['input_ids'].shape[0]

        # Process VLM inputs to tokens
        t0 = time.time()
        inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids = \
            self._process_vlm_inputs_to_tokens(vlm_inputs, B)
        timing['process_inputs'] = time.time() - t0

        # Check if labels are present for VLM loss
        has_labels = isinstance(vlm_inputs, dict) and 'labels' in vlm_inputs
        labels = vlm_inputs['labels'].to(self.device) if has_labels else None

        # Set up VLM forward kwargs
        vlm_kwargs = {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'past_key_values': None,
            'use_cache': False,
            'output_attentions': False,
            'output_hidden_states': True,  # KEY: get all layer outputs
            'return_dict': True
        }

        # Add labels for loss computation if available
        if labels is not None:
            vlm_kwargs['labels'] = labels

        # Add DeepStack parameters for Qwen3-VL
        if visual_pos_masks is not None:
            vlm_kwargs['visual_pos_masks'] = visual_pos_masks
        if deepstack_image_embeds is not None:
            vlm_kwargs['deepstack_visual_embeds'] = deepstack_image_embeds

        # Forward through VLM language model (for MoT hidden states)
        t1 = time.time()
        vlm_output = self.vlm_model.model.language_model(**vlm_kwargs)
        timing['lm_forward'] = time.time() - t1

        # Extract VLM loss (language_model doesn't return loss directly)
        vlm_loss = None

        # Compute loss manually if labels are provided
        if labels is not None:
            # Get lm_head from VLM model (not language_model)
            lm_head = self.vlm_model.lm_head
            # Get last hidden state
            last_hidden_state = vlm_output.last_hidden_state  # [B, seq_len, hidden_dim]

            # Compute logits
            logits = torch.nn.functional.linear(last_hidden_state, lm_head.weight, lm_head.bias)

            # Cross-entropy loss
            from torch.nn import CrossEntropyLoss
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            vlm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Return all layer hidden states
        # hidden_states is a tuple: (embeddings, layer_0, layer_1, ..., layer_27)
        # We skip the embeddings (index 0) and return the layer outputs
        hidden_states = vlm_output.hidden_states  # Tuple of (num_layers + 1) tensors

        t2 = time.time()
        result = list(hidden_states[1:])  # Return only layer outputs: [0, ..., 27] = 28 layers

        # Map 28 VLM layers to 30 WAN layers
        # WAN Layer 28 uses VLM Layer 26 (second to last)
        # WAN Layer 29 uses VLM Layer 27 (last layer)
        result.append(result[-2].clone())  # VLM Layer 26 -> WAN Layer 28
        result.append(result[-1].clone())  # VLM Layer 27 -> WAN Layer 29
        timing['map_layers'] = time.time() - t2

        assert len(result) == 30, f"Expected 30 layers, got {len(result)}"

        # Log timing breakdown (only once)
        if not hasattr(self, '_logged_extract_timing'):
            logger.info(f"Qwen3VLWanModule.extract_per_layer_features breakdown:")
            logger.info(f"  Process inputs (image + text): {timing['process_inputs']*1000:.1f}ms")
            logger.info(f"  Language model forward (28 layers): {timing['lm_forward']*1000:.1f}ms")
            logger.info(f"  Map layers (28 -> 30): {timing['map_layers']*1000:.1f}ms")
            self._logged_extract_timing = True

        return result, vlm_loss

    def prepare_vlm_inputs_with_action(
        self,
        vlm_inputs: Dict,
        actions: Optional[torch.Tensor] = None,
        fast_tokenizer=None
    ) -> Dict:
        """
        Prepare VLM inputs with action tokens for VLM action loss computation.

        This appends action tokens as the assistant response and creates labels
        where only action token positions are unmasked.

        Args:
            vlm_inputs: Original VLM inputs dict with keys:
                - input_ids: [B, seq_len]
                - attention_mask: [B, seq_len]
                - pixel_values: image features
                - image_grid_thw: image grid dimensions
            actions: Action tensors [B, T, action_dim] or None
            fast_tokenizer: FastActionTokenizer instance for action discretization

        Returns:
            Dict: VLM inputs with action tokens and labels added
        """
        # If action loss is disabled or no actions provided, return unchanged
        if actions is None or fast_tokenizer is None:
            if self.config.enable_vlm_action_loss and actions is not None:
                logger.warning("Actions provided but VLM action loss is disabled")
            return vlm_inputs

        # Convert actions to VLM format
        try:
            action_token_ids = self._convert_actions_to_vlm_tokens(
                actions, fast_tokenizer
            )
        except Exception as e:
            logger.warning(f"Failed to convert actions to VLM tokens: {e}")
            return vlm_inputs

        # Use external tokenizer (loaded in main model)
        if self.tokenizer is None:
            logger.warning("No tokenizer available - cannot prepare VLM inputs with action tokens")
            return vlm_inputs

        tokenizer = self.tokenizer
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        # Append action tokens to input_ids
        original_input_ids = vlm_inputs['input_ids']  # [B, seq_len]
        B, original_seq_len = original_input_ids.shape

        # Find max action tokens length in batch
        max_action_len = max([len(tokens) for tokens in action_token_ids])

        # Create padded action tokens tensor
        action_padded = torch.full(
            (B, max_action_len),
            pad_token_id,
            dtype=original_input_ids.dtype,
            device=original_input_ids.device
        )

        # Fill action tokens (right-aligned, padded on left)
        for i, tokens in enumerate(action_token_ids):
            if len(tokens) > 0:
                start_idx = max_action_len - len(tokens)
                action_padded[i, start_idx:] = torch.tensor(
                    tokens,
                    dtype=original_input_ids.dtype,
                    device=original_input_ids.device
                )

        # Concatenate: [B, original_seq_len + max_action_len]
        input_ids_with_action = torch.cat(
            [original_input_ids, action_padded],
            dim=1
        )

        # Update attention mask
        original_attn_mask = vlm_inputs['attention_mask']
        original_attn_dtype = original_attn_mask.dtype
        action_attn_mask = torch.ones(  # Action tokens should participate in attention for loss computation
            (B, max_action_len),
            dtype=original_attn_dtype,
            device=original_attn_mask.device
        )
        attention_mask_with_action = torch.cat(
            [original_attn_mask, action_attn_mask],
            dim=1
        )

        # Create labels: mask everything except action tokens
        labels = torch.full_like(input_ids_with_action, -100)
        for i, tokens in enumerate(action_token_ids):
            if len(tokens) > 0:
                start_idx = max_action_len - len(tokens)
                labels[i, original_seq_len + start_idx:] = action_padded[i, start_idx:]

        # Update vlm_inputs
        vlm_inputs_updated = {
            'input_ids': input_ids_with_action,
            'attention_mask': attention_mask_with_action,
            'labels': labels,
            'pixel_values': vlm_inputs['pixel_values'],
            'image_grid_thw': vlm_inputs['image_grid_thw']
        }

        return vlm_inputs_updated

    def _convert_actions_to_vlm_tokens(
        self,
        actions: torch.Tensor,
        fast_tokenizer
    ) -> List[List[int]]:
        """
        Convert continuous actions to VLM action token IDs using fast tokenizer.

        Pipeline:
        1. Actions -> fast_tokenizer.encoder_action2fastoken() -> BPE token IDs
        2. BPE token IDs -> fast_tokenizer.map_bpe_tokens_to_vlm_action() -> <robot_action_X> string
        3. <robot_action_X> string -> tokenizer.encode() -> VLM token IDs

        Args:
            actions: Action tensors [B, T, action_dim]
            fast_tokenizer: FastActionTokenizer instance

        Returns:
            List[List[int]]: VLM action token IDs for each sample
        """
        B, T, action_dim = actions.shape

        # Convert to numpy
        actions_np = actions.cpu().float().numpy()

        batch_vlm_token_ids = []
        for i in range(B):
            action_i = actions_np[i]  # [T, action_dim]

            try:
                # Step 1: Use fast tokenizer to get BPE token IDs
                bpe_token_ids = fast_tokenizer.encoder_action2fastoken([action_i])[0]

                # Step 2: Map BPE token IDs to VLM action format
                vlm_action_string = fast_tokenizer.map_bpe_tokens_to_vlm_action(bpe_token_ids)

                # Step 3: Convert to VLM token IDs
                token_ids = self.tokenizer.encode(
                    vlm_action_string,
                    add_special_tokens=False
                )

                batch_vlm_token_ids.append(token_ids)

            except Exception as e:
                import traceback
                logger.warning(f"Failed to convert actions to VLM tokens (sample {i}): {e}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                return []

        return batch_vlm_token_ids

    def _process_vlm_inputs_to_tokens(
        self,
        vlm_inputs,
        B: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[list], torch.Tensor]:
        """Convert VLM inputs to tokens.

        Returns:
            Tuple of (inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids)
        """
        import time
        timing = {}

        # Handle both list format and batched dict format
        t0 = time.time()
        if isinstance(vlm_inputs, list):
            # List format: do padding and batching
            input_ids_list = [vlm_input['input_ids'] for vlm_input in vlm_inputs]
            attention_mask_list = [vlm_input.get('attention_mask') for vlm_input in vlm_inputs]
            pixel_values_list = [vlm_input.get('pixel_values') for vlm_input in vlm_inputs]
            image_grid_thw_list = [vlm_input.get('image_grid_thw') for vlm_input in vlm_inputs]

            # Pad input_ids and attention_mask to same length
            max_seq_len = max(ids.shape[1] for ids in input_ids_list)
            padded_input_ids = []
            padded_attention_masks = []

            for ids, mask in zip(input_ids_list, attention_mask_list):
                if ids.shape[1] < max_seq_len:
                    padding_size = max_seq_len - ids.shape[1]
                    id_padding = torch.zeros(ids.shape[0], padding_size, dtype=ids.dtype, device=ids.device)
                    padded_ids = torch.cat([ids, id_padding], dim=1)
                    mask_padding = torch.zeros(mask.shape[0], padding_size, dtype=mask.dtype, device=mask.device)
                    padded_mask = torch.cat([mask, mask_padding], dim=1)
                else:
                    padded_ids = ids
                    padded_mask = mask
                padded_input_ids.append(padded_ids)
                padded_attention_masks.append(padded_mask)

            # Batch process
            input_ids_batch = torch.cat(padded_input_ids, dim=0).to(self.device)
            attention_mask_batch = torch.cat(padded_attention_masks, dim=0).to(self.device)
            pixel_values_batch = torch.cat([pv.to(self.device) for pv in pixel_values_list], dim=0)
            image_grid_thw_batch = torch.cat([igt.to(self.device) for igt in image_grid_thw_list], dim=0)
        else:
            # Batched dict format: already padded
            input_ids_batch = vlm_inputs['input_ids'].to(self.device)
            attention_mask_batch = vlm_inputs['attention_mask'].to(self.device)
            pixel_values_batch = vlm_inputs['pixel_values'].to(self.device)
            image_grid_thw_batch = vlm_inputs['image_grid_thw'].to(self.device)
        timing['padding'] = time.time() - t0

        # Get input embeddings
        t1 = time.time()
        inputs_embeds = self.vlm_model.get_input_embeddings()(input_ids_batch)
        timing['text_embeds'] = time.time() - t1

        # Process images (with no_grad for vision encoder, train only language model)
        t2 = time.time()
        with torch.no_grad():
            image_embeds, deepstack_image_embeds = self.vlm_model.get_image_features(
                pixel_values_batch, image_grid_thw_batch
            )
        timing['image_features'] = time.time() - t2

        t3 = time.time()
        image_embeds = torch.cat(image_embeds, dim=0).to(self.device, self.dtype)
        timing['image_cat_dtype'] = time.time() - t3

        # Insert image embeddings
        t4 = time.time()
        image_mask, _ = self.vlm_model.model.get_placeholder_mask(
            input_ids_batch, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        timing['image_insert'] = time.time() - t4

        visual_pos_masks = image_mask[..., 0]  # [B, seq_len] - visual positions only

        # Compute position_ids
        t5 = time.time()
        position_ids, _rope_deltas = self.vlm_model.model.get_rope_index(
            input_ids=input_ids_batch,
            image_grid_thw=image_grid_thw_batch,
            video_grid_thw=None,  # No video in current implementation
            attention_mask=attention_mask_batch
        )
        timing['rope_index'] = time.time() - t5

        # Log timing breakdown (only once)
        if not hasattr(self, '_logged_process_timing'):
            logger.info(f"Qwen3VLWanModule._process_vlm_inputs_to_tokens breakdown:")
            logger.info(f"  Padding: {timing['padding']*1000:.1f}ms")
            logger.info(f"  Text embeddings: {timing['text_embeds']*1000:.1f}ms")
            logger.info(f"  Image features get_image_features: {timing['image_features']*1000:.1f}ms")
            logger.info(f"    - image_cat_dtype: {timing['image_cat_dtype']*1000:.1f}ms")
            logger.info(f"  Image insert (masked_scatter): {timing['image_insert']*1000:.1f}ms")
            logger.info(f"  RoPE index: {timing['rope_index']*1000:.1f}ms")
            self._logged_process_timing = True

        return inputs_embeds, attention_mask_batch, visual_pos_masks, deepstack_image_embeds, position_ids

    def process_ffn(self, vlm_tokens: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Process Qwen3-VL after MoT attention.

        The residual connection is already applied in process_joint_attention():
            vlm_tokens = vlm_tokens + vlm_out

        Unlike Video and Action modules, Qwen3-VL's FFN is handled internally
        by the Qwen3-VL model itself (via extract_per_layer_features).
        We just need to return the tokens unchanged.

        Args:
            vlm_tokens: Features after MoT attention with residual [B, seq_len, 2048]
            layer_idx: Which layer block to use (unused, kept for interface consistency)

        Returns:
            The same tokens [B, seq_len, 2048]
        """
        # Direct return - residual is already applied in process_joint_attention
        # Qwen3-VL's internal FFN is already computed via extract_per_layer_features
        return vlm_tokens
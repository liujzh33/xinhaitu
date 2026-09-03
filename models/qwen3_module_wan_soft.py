# Qwen3-VL Module for Direct MoT Participation with WAN
# Each layer has its own QKV projection to the unified WAN head space
# VLM has 28 layers, WAN has 30 layers - we map VLM Layer 26, 27 to WAN Layer 28, 29

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging

import sys
from pathlib import Path

BAK_ROOT = str((Path(__file__).parent.parent / "bak").resolve())
if str(BAK_ROOT) not in sys.path:
    sys.path.insert(0, str(BAK_ROOT))

from wan.modules.model import WanLayerNorm, WanRMSNorm

logger = logging.getLogger(__name__)


class LearnableAlignment(nn.Module):
    """
    Learnable Layer Alignment - Soft Alignment for VLM layers to WAN layers.

    Instead of hard-coded layer mapping (VLM 0-27 -> WAN 0-27, VLM 26->28, VLM 27->29),
    this module learns a soft alignment matrix where each WAN layer is a weighted combination
    of all VLM layers using softmax attention.

    The alignment is computed as:
        aligned_features[i] = sum_j softmax(logits[i, j] * temperature) * vlm_features[j]

    This allows the model to learn optimal layer correspondence during training.

    Args:
        vlm_layers: Number of VLM layers (e.g., 28 for Qwen3-VL-2B)
        wan_layers: Number of WAN layers (e.g., 30 for WAN 2.2 5B)
        temperature: Temperature for softmax, lower = sparser, higher = smoother
        temperature_decay: If True, temperature decays during training (starts high, ends low)
        init_identity: If True, initialize with near-identity mapping
    """

    def __init__(
        self,
        vlm_layers: int = 28,
        wan_layers: int = 30,
        temperature: float = 1.0,
        temperature_decay: bool = False,
        init_identity: bool = False
    ):
        super().__init__()
        self.vlm_layers = vlm_layers
        self.wan_layers = wan_layers
        self.temperature_decay = temperature_decay
        self.base_temperature = temperature

        # Learnable alignment logits: [wan_layers, vlm_layers]
        # Initialize with small random values
        alignment_logits = torch.zeros(wan_layers, vlm_layers)
        if init_identity:
            # Initialize with identity bias for layers that map directly
            # For layers 0-27: VLM 0-27 -> WAN 0-27
            # For layers 28-29: bias toward VLM 26-27
            for i in range(min(wan_layers, vlm_layers)):
                alignment_logits[i, i] = 2.0
            if wan_layers > vlm_layers:
                # Extra WAN layers bias toward last VLM layers
                alignment_logits[vlm_layers, vlm_layers - 2] = 2.0
                alignment_logits[vlm_layers + 1, vlm_layers - 1] = 2.0

        self.alignment_logits = nn.Parameter(alignment_logits)
        self.register_buffer('current_temperature', torch.tensor(temperature))

    def forward(self, vlm_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Align VLM features to WAN layers using learnable soft alignment.

        Args:
            vlm_features: List of VLM layer features [layer_0, ..., layer_27]
                         Each tensor: [B, seq_len, vlm_dim]

        Returns:
            List of aligned features for WAN layers: [aligned_0, ..., aligned_29]
            Each tensor: [B, seq_len, vlm_dim]
        """
        # Stack VLM features: [vlm_layers, B, seq_len, vlm_dim]
        stacked_vlm = torch.stack(vlm_features, dim=0)  # [L, B, S, D]

        # Compute softmax weights: [wan_layers, vlm_layers]
        weights = torch.softmax(self.alignment_logits / self.current_temperature, dim=-1)

        # Compute weighted sum for each WAN layer
        # weights @ stacked_vlm: [wan_layers, B, seq_len, vlm_dim]
        aligned_features = torch.einsum('ij,jbsd->ibsd', weights, stacked_vlm)

        # Convert back to list
        result = [aligned_features[i] for i in range(self.wan_layers)]

        return result

    def update_temperature(self, progress: float, min_temp: float = 0.1):
        """
        Update temperature based on training progress (0.0 to 1.0).

        When temperature_decay=True, temperature decreases during training:
        - Early training: high temperature (smooth, broad attention)
        - Late training: low temperature (sparse, focused attention)

        Args:
            progress: Training progress (0.0 to 1.0)
            min_temp: Minimum temperature at the end of training
        """
        if self.temperature_decay:
            new_temp = self.base_temperature * (1 - progress) + min_temp * progress
            self.current_temperature.copy_(new_temp)

    def get_alignment_weights(self) -> torch.Tensor:
        """Get current alignment weights (after softmax)."""
        return torch.softmax(self.alignment_logits / self.current_temperature, dim=-1)

    def get_alignment_entropy(self) -> torch.Tensor:
        """
        Compute entropy of alignment weights (measure of ambiguity).

        High entropy = smooth alignment (attending to many layers)
        Low entropy = sparse alignment (attending to few layers)

        Returns:
            Average entropy across all WAN layers
        """
        weights = self.get_alignment_weights()
        entropy = -(weights * torch.log(weights + 1e-10)).sum(dim=-1)
        return entropy.mean()


@dataclass
class Qwen3VLWanConfig:
    """Configuration for Qwen3-VL Expert model with WAN backbone."""
    vlm_dim: int = 2048              # Qwen3-VL-2B hidden size
    head_dim: int = 128              # Head dimension (for WAN 5B)
    num_heads: int = 24              # Number of heads (for WAN 5B)
    num_layers: int = 30             # Number of layers (matches WAN)
    eps: float = 1e-5                # Layer norm epsilon

    # Learnable Layer Alignment settings
    use_learnable_alignment: bool = False  # Whether to use learnable soft alignment
    alignment_temperature: float = 1.0     # Temperature for softmax alignment
    alignment_temperature_decay: bool = False  # Whether to decay temperature during training
    alignment_init_identity: bool = False  # Initialize with near-identity mapping


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

    def __init__(self, vlm_model, config: Qwen3VLWanConfig, dtype, device):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.device = device

        # VLM model reference
        self.vlm_model = vlm_model

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

        # Learnable layer alignment (optional)
        if config.use_learnable_alignment:
            self.alignment = LearnableAlignment(
                vlm_layers=28,  # Qwen3-VL-2B has 28 layers
                wan_layers=config.num_layers,  # WAN has 30 layers
                temperature=config.alignment_temperature,
                temperature_decay=config.alignment_temperature_decay,
                init_identity=config.alignment_init_identity
            )
            self.alignment.to(device=device, dtype=dtype)
            logger.info("Learnable layer alignment enabled")
            logger.info(f"  Temperature: {config.alignment_temperature}")
            logger.info(f"  Temperature decay: {config.alignment_temperature_decay}")
            logger.info(f"  Init identity: {config.alignment_init_identity}")
        else:
            self.alignment = None
            logger.info("Using hard-coded layer mapping")

        self.blocks.to(device=device, dtype=dtype)
        logger.info(f"Qwen3VLWanModule initialized with {config.num_layers} layers (WAN)")
        logger.info(f"  VLM dim: {config.vlm_dim}, WAN head space: {config.head_dim * config.num_heads}")
        if not config.use_learnable_alignment:
            logger.info(f"  Layer mapping: VLM 0-27 -> WAN 0-27, VLM 26->WAN 28, VLM 27->WAN 29")

    def extract_per_layer_features(
        self,
        vlm_inputs
    ) -> List[torch.Tensor]:
        """
        Extract Qwen3-VL hidden states from ALL layers for per-layer MoT.

        VLM has 28 layers, but WAN has 30 layers.
        We map: VLM Layer 26 -> WAN Layer 28, VLM Layer 27 -> WAN Layer 29

        Args:
            vlm_inputs: VLM inputs (list of dicts or dict of tensors)

        Returns:
            List of hidden states, 30 layers total: [layer_0, ..., layer_27, layer_26, layer_27]
            Each tensor: [B, seq_len, 2048]
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

        # Add DeepStack parameters for Qwen3-VL
        if visual_pos_masks is not None:
            vlm_kwargs['visual_pos_masks'] = visual_pos_masks
        if deepstack_image_embeds is not None:
            vlm_kwargs['deepstack_visual_embeds'] = deepstack_image_embeds

        # Forward through VLM (with grad based on training mode)
        t1 = time.time()
        vlm_output = self.vlm_model.model.language_model(**vlm_kwargs)
        timing['lm_forward'] = time.time() - t1

        # Return all layer hidden states
        # hidden_states is a tuple: (embeddings, layer_0, layer_1, ..., layer_27)
        # We skip the embeddings (index 0) and return the layer outputs
        hidden_states = vlm_output.hidden_states  # Tuple of (num_layers + 1) tensors

        t2 = time.time()
        vlm_features = list(hidden_states[1:])  # Return only layer outputs: [0, ..., 27] = 28 layers

        # Apply layer alignment
        if self.alignment is not None:
            # Use learnable soft alignment: 28 VLM layers -> 30 WAN layers
            result = self.alignment(vlm_features)
            timing['soft_alignment'] = time.time() - t2
        else:
            # Hard-coded mapping: 28 VLM layers -> 30 WAN layers
            # WAN Layer 28 uses VLM Layer 26 (second to last)
            # WAN Layer 29 uses VLM Layer 27 (last layer)
            result = vlm_features.copy()
            result.append(result[-2].clone())  # VLM Layer 26 -> WAN Layer 28
            result.append(result[-1].clone())  # VLM Layer 27 -> WAN Layer 29
            timing['map_layers'] = time.time() - t2

        assert len(result) == 30, f"Expected 30 layers, got {len(result)}"

        # Log timing breakdown (only once)
        if not hasattr(self, '_logged_extract_timing'):
            logger.info(f"Qwen3VLWanModule.extract_per_layer_features breakdown:")
            logger.info(f"  Process inputs (image + text): {timing['process_inputs']*1000:.1f}ms")
            logger.info(f"  Language model forward (28 layers): {timing['lm_forward']*1000:.1f}ms")
            if self.alignment is not None:
                logger.info(f"  Soft alignment (28 -> 30): {timing['soft_alignment']*1000:.1f}ms")
            else:
                logger.info(f"  Map layers (28 -> 30): {timing['map_layers']*1000:.1f}ms")
            self._logged_extract_timing = True

        return result

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

    def update_alignment_temperature(self, progress: float, min_temp: float = 0.1):
        """
        Update alignment temperature based on training progress.

        Args:
            progress: Training progress (0.0 to 1.0)
            min_temp: Minimum temperature at the end of training
        """
        if self.alignment is not None:
            self.alignment.update_temperature(progress, min_temp)

    def get_alignment_weights(self) -> Optional[torch.Tensor]:
        """
        Get current alignment weights (after softmax).

        Returns:
            Alignment matrix [wan_layers, vlm_layers], or None if alignment is disabled
        """
        if self.alignment is not None:
            return self.alignment.get_alignment_weights()
        return None

    def get_alignment_entropy(self) -> Optional[float]:
        """
        Get current alignment entropy.

        Returns:
            Average entropy across all WAN layers, or None if alignment is disabled
        """
        if self.alignment is not None:
            return self.alignment.get_alignment_entropy().item()
        return None
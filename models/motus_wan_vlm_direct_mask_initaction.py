# Motus-WAN-VLM Direct - Modular Architecture
# Three-modal UniDiffuser: Video Model (WAN) + Action Expert + VLM (Direct MoT)
# VLM 28 layers -> WAN 30 layers (Layer 26, 27 reused for WAN Layer 28, 29)

import sys
import json
import time
import torch
import logging
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

BAK_ROOT = str((Path(__file__).parent.parent / "bak").resolve())
if BAK_ROOT not in sys.path:
    sys.path.insert(0, BAK_ROOT)

from utils.common import get_t_distribution
from wan.modules.model_mask import sinusoidal_embedding_1d
from transformers import Qwen3VLForConditionalGeneration, AutoConfig
from .wan_model_mask import WanVideoModel
from .action_expert_initaction import ActionExpert, ActionExpertConfig
from .qwen3_module_wan import Qwen3VLWanModule, Qwen3VLWanConfig
# Add Flow-Matching schedulers
from wan.utils.fm import FlowMatchScheduler
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

logger = logging.getLogger(__name__)


@dataclass
class MotusWanVlmDirectMaskConfig:
    """Configuration for MotusWanVlmDirect with asymmetric attention mask."""
    # Video model settings (WAN 5B)
    wan_checkpoint_path: str = ""
    vae_path: str = ""
    wan_config_path: str = ""
    video_precision: str = "bfloat16"

    # VLM settings (Qwen3-VL-2B, trainable)
    vlm_checkpoint_path: str = ""

    # Qwen3-VL Expert settings (per-layer QKV projections)
    vlm_dim: int = 2048              # Qwen3-VL-2B hidden size
    qwen3_expert_head_dim: int = 128              # Head dimension (for WAN 5B)
    qwen3_expert_num_heads: int = 24              # Number of heads (for WAN 5B)
    qwen3_expert_num_layers: int = 30             # Number of layers (matches WAN)
    qwen3_expert_norm_eps: float = 1e-5           # Layer norm epsilon

    # Action expert settings
    num_layers: int = 30
    action_state_dim: int = 14
    action_dim: int = 14
    action_expert_dim: int = 1024           # Configurable hidden dimension
    action_expert_ffn_dim_multiplier: int = 4  # FFN dimension multiplier
    action_expert_norm_eps: float = 1e-6    # Layer norm epsilon for Action Expert
    action_pretrained_path: str = ""  # Path to pre-processed Action Expert backbone .pt file

    # Sampling settings
    global_downsample_rate: int = 3     # Global downsampling rate
    video_action_freq_ratio: int = 4    # Video:Action frequency ratio
    num_video_frames: int = 4           # Number of video frames to predict

    # Video dimensions
    video_height: int = 512             # Input video height
    video_width: int = 512              # Input video width

    # Training settings
    batch_size: int = 8

    # Training mode
    training_mode: str = 'finetune'  # 'pretrain' or 'finetune'

    # Loss weights
    video_loss_weight: float = 1.0
    action_loss_weight: float = 1.0

    # Control whether to load pretrained WAN/VLM backbones
    load_pretrained_backbones: Optional[bool] = None

    # VLM frozen setting
    vlm_frozen: bool = False  # VLM is trainable by default

    def __post_init__(self):
        """Calculate derived parameters."""
        # Action chunk size is determined by global downsample rate and frequency ratio
        self.action_chunk_size = self.num_video_frames * self.video_action_freq_ratio


class VideoModule(nn.Module):
    """Video processing module - handles WAN + T5 operations."""

    def __init__(self, video_model, dtype, device, grid_sizes):
        super().__init__()
        self.video_model = video_model
        self.dtype = dtype
        self.device = device
        self.grid_sizes = grid_sizes

    def prepare_input(self, noisy_video_latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare video tokens from pre-processed noisy latent."""
        # Through patch_embedding: 48 -> 3072 channels
        video_patched = self.video_model.wan_model.patch_embedding(noisy_video_latent)
        # Flatten and convert to tokens
        video_features = video_patched.flatten(2).transpose(1, 2)
        return video_features

    def preprocess_t5_embeddings(self, language_embeddings) -> torch.Tensor:
        """Pre-process T5 embeddings once for all layers."""
        text_len = self.video_model.wan_model.text_len  # 512
        if isinstance(language_embeddings, list):
            padded_embeddings = []
            for emb in language_embeddings:
                if emb.shape[0] <= text_len:
                    padded = torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])])
                else:
                    padded = emb[:text_len]
                padded_embeddings.append(padded)
            t5_context_raw = torch.stack(padded_embeddings, dim=0)
        else:
            t5_context_raw = language_embeddings
        # Convert via text_embedding layer (4096 -> 3072)
        t5_context = self.video_model.wan_model.text_embedding(t5_context_raw)
        return t5_context

    def get_time_embedding(self, t_video: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get WAN's time embedding using WAN's own weights."""
        if t_video.dim() == 1:
            t_video = t_video.unsqueeze(1).expand(t_video.size(0), seq_len)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t_video.size(0)
            t_flat = t_video.flatten()

            t_emb = self.video_model.wan_model.time_embedding(
                sinusoidal_embedding_1d(self.video_model.wan_model.freq_dim, t_flat).unflatten(0, (bt, seq_len)).float()
            )
            t_emb_proj = self.video_model.wan_model.time_projection(t_emb).unflatten(2, (6, 3072))
            assert t_emb.dtype == torch.float32 and t_emb_proj.dtype == torch.float32

        return t_emb, t_emb_proj

    def process_cross_attention(self, video_tokens: torch.Tensor, video_adaln_params: torch.Tensor,
                               layer_idx: int, processed_t5_context: torch.Tensor) -> torch.Tensor:
        """Process WAN cross attention with pre-processed T5 context."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        context_lens = None
        cross_out = wan_layer.cross_attn(wan_layer.norm3(video_tokens), processed_t5_context, context_lens)
        return video_tokens + cross_out

    def compute_adaln_modulation(self, video_adaln_params: torch.Tensor, layer_idx: int) -> tuple:
        """Compute AdaLN modulation parameters for WAN (6 components)."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        with torch.amp.autocast('cuda', dtype=torch.float32):
            modulation = (
                wan_layer.modulation.unsqueeze(0)
                + video_adaln_params
            ).chunk(6, dim=2)
        return modulation

    def process_ffn(self, video_tokens: torch.Tensor, video_adaln_modulation: tuple, layer_idx: int) -> torch.Tensor:
        """Process WAN FFN with proper AdaLN modulation."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]

        # AdaLN params
        v_mod = video_adaln_modulation

        # WAN FFN with AdaLN (params 3,4,5 for FFN: α3, β3, γ3)
        ffn_input = wan_layer.norm2(video_tokens).float() * (1 + v_mod[4].squeeze(2)) + v_mod[3].squeeze(2)
        ffn_out = wan_layer.ffn(ffn_input)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            return video_tokens + ffn_out * v_mod[5].squeeze(2)

    def apply_output_head(self, video_tokens: torch.Tensor, video_time_emb: torch.Tensor) -> torch.Tensor:
        """Apply WAN's head + unpatchify for final video output."""
        x = self.video_model.wan_model.head(video_tokens, video_time_emb)
        x = self.video_model.wan_model.unpatchify(x, self.grid_sizes)
        return torch.stack([u.float() for u in x], dim=0)

    def process_joint_attention(
        self,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        vlm_tokens: torch.Tensor,
        video_adaln_modulation: tuple,
        action_adaln_modulation: tuple,
        layer_idx: int,
        action_block: nn.Module,
        vlm_block: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Trimodal joint self-attention: WAN + Action + VLM via WAN self-attn (MoT)."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]

        # AdaLN params (already computed)
        v_mod = video_adaln_modulation
        a_mod = action_adaln_modulation

        # Pre-attn normalization with AdaLN
        norm_video = wan_layer.norm1(video_tokens).float() * (1 + v_mod[1].squeeze(2)) + v_mod[0].squeeze(2)
        norm_action = action_block.norm1(action_tokens) * (1 + a_mod[1].squeeze(2)) + a_mod[0].squeeze(2)

        # Get dimensions
        B, L_v, C = norm_video.shape
        L_a = norm_action.shape[1]
        n = self.video_model.wan_model.num_heads
        d = C // n

        # Action heads for WAN space (1024 -> 24*128)
        a_qkv = torch.einsum("BTD,KNDE->KBTNE", norm_action, action_block.wan_action_qkv)
        a_q_h, a_k_h, a_v_h = a_qkv[0], a_qkv[1], a_qkv[2]
        a_q = action_block.wan_action_norm_q(a_q_h.flatten(-2)).view(B, L_a, n, d)
        a_k = action_block.wan_action_norm_k(a_k_h.flatten(-2)).view(B, L_a, n, d)
        a_v = a_v_h.view(B, L_a, n, d)

        # VLM processing
        norm_vlm = vlm_block.norm1(vlm_tokens)
        L_vlm = norm_vlm.shape[1]

        # VLM heads for WAN space (2048 -> 24*128)
        v_qkv = torch.einsum("BTD,KNDE->KBTNE", norm_vlm, vlm_block.wan_vlm_qkv)
        v_q_h, v_k_h, v_v_h = v_qkv[0], v_qkv[1], v_qkv[2]
        v_q = vlm_block.wan_vlm_norm_q(v_q_h.flatten(-2)).view(B, L_vlm, n, d)
        v_k = vlm_block.wan_vlm_norm_k(v_k_h.flatten(-2)).view(B, L_vlm, n, d)
        v_v = v_v_h.view(B, L_vlm, n, d)

        # Meta info for WAN attention
        seq_lens = torch.full((B,), L_v + L_a + L_vlm, dtype=torch.long, device=self.device)
        freqs = self.video_model.wan_model.freqs
        if freqs.device != self.device:
            freqs = freqs.to(self.device)

        # ==================== 构建非对称 attention mask ====================
        # Attention pattern: Action看所有, Video/VLM只自注意力
        L_total = L_v + L_a + L_vlm
        attn_mask = torch.ones((B, L_total, L_total), dtype=torch.bool, device=self.device)

        # # Video 不能看 Action 和 VLM (只看 Video自注意力)
        attn_mask[:, :L_v, L_v:] = False

        # # VLM 不能看 Video 和 Action (只看 VLM自注意力)
        attn_mask[:, L_v + L_a:, :L_v + L_a] = False

        # Action 看所有（默认为True，无需修改）
        # =================================================================

        # Print mask for verification (only once at first layer of first step)
        if layer_idx == 0 and not hasattr(self, '_mask_printed'):
            logger.info("=" * 60)
            logger.info(f"Attention mask (L_v={L_v}, L_a={L_a}, L_vlm={L_vlm}):")
            logger.info("Rows=Query, Cols=Key, 1=allow, 0=mask")
            # Print compact representation
            mask_np = attn_mask[0].cpu().numpy().astype(int)
            logger.info(f"Video rows (first 5): {mask_np[:L_v, :5]}")
            logger.info(f"Video rows (Action/VLM cols): {mask_np[:L_v, L_v:L_v+5]}")
            logger.info(f"Action rows (Video cols, first 5): {mask_np[L_v:L_v+1, :5]}")
            logger.info(f"VLM rows (Video cols, first 5): {mask_np[L_v+L_a:, :5]}")
            logger.info("V=Video, A=Action, V=VLM")
            logger.info("Expected: Video→Video✅, Action→all✅, VLM→VLM✅, cross:❌")
            logger.info("=" * 60)
            self._mask_printed = True

        # Call WAN self-attn with trimodal MoT and custom mask
        y, action_out_h, vlm_out_h = wan_layer.self_attn(
            norm_video, seq_lens, self.grid_sizes, freqs,
            action_q=a_q, action_k=a_k, action_v=a_v,
            vlm_q=v_q, vlm_k=v_k, vlm_v=v_v,
            attn_mask=attn_mask  # 非对称 attention mask
        )

        # Project VLM output back to VLM dimension (3072 -> 2048)
        vlm_out = vlm_block.wan_vlm_o(vlm_out_h.flatten(2))

        # Project back and residual connections
        action_out = action_block.wan_action_o(action_out_h.flatten(2))
        video_tokens = video_tokens + y * v_mod[2].squeeze(2)
        action_tokens = action_tokens + action_out * a_mod[2].squeeze(2)
        vlm_tokens = vlm_tokens + vlm_out  # Regular residual connection

        return video_tokens, action_tokens, vlm_tokens


class ActionModule(nn.Module):
    """Action processing module - handles Action Expert + joint attentions + masks."""

    def __init__(self, action_expert: ActionExpert, config, video_model, vlm_model, dtype, device):
        super().__init__()
        self.action_expert = action_expert
        self.config = config
        self.video_model = video_model  # For accessing WAN weights
        self.vlm_model = vlm_model      # For accessing VLM weights
        self.dtype = dtype
        self.device = device

    def get_time_embedding(self, t: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get action time embedding."""
        if t.dim() == 1:
            t = t.unsqueeze(1).expand(t.size(0), seq_len)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t_flat = t.flatten()

            # Create sinusoidal embedding (same pattern as VideoModule)
            a_e = self.action_expert.time_embedding(
                sinusoidal_embedding_1d(self.action_expert.freq_dim, t_flat).unflatten(0, (bt, seq_len)).float()
            )  # [B, seq_len, freq_dim]

            # Project to AdaLN parameters (6 params: 3 for WAN-Action joint attn + 3 for FFN)
            a_e0 = self.action_expert.time_projection(a_e).unflatten(2, (6, self.config.action_expert_dim))  # [B, seq_len, 6, dim]

            assert a_e.dtype == torch.float32 and a_e0.dtype == torch.float32

        return a_e, a_e0  # (basic_emb, adaln_params)

    def compute_adaln_modulation(self, action_adaln_params: torch.Tensor, layer_idx: int) -> tuple:
        """Compute AdaLN modulation parameters for 6 components (3 for WAN-Action joint attn + 3 for FFN)."""
        action_layer = self.action_expert.blocks[layer_idx]
        with torch.amp.autocast('cuda', dtype=torch.float32):
            modulation = (
                action_layer.modulation.unsqueeze(0)
                + action_adaln_params
            ).chunk(6, dim=2)
        return modulation

    def process_ffn(self, action_tokens: torch.Tensor, action_adaln_modulation: tuple, layer_idx: int) -> torch.Tensor:
        """Process Action Expert FFN with AdaLN modulation."""
        action_block = self.action_expert.blocks[layer_idx]

        # AdaLN params
        a_mod = action_adaln_modulation

        # Apply FFN with AdaLN modulation (params 3,4,5 for FFN: α3, β3, γ3)
        ffn_input = action_block.norm2(action_tokens).float() * (1 + a_mod[4].squeeze(2)) + a_mod[3].squeeze(2)
        ffn_out = action_block.ffn(ffn_input)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            action_tokens = action_tokens + ffn_out * a_mod[5].squeeze(2)
        return action_tokens


class MotusWanVlmDirectMask(nn.Module):
    """
    Modular Three-modal UniDiffuser with VGM (WAN), VLM (Direct MoT), and Action modules.
    Uses asymmetric attention mask: Action sees all, Video/VLM only self-attention.

    Attention Matrix:
        Query \ Key  | Video | Action | VLM
        -------------|-------|--------|-----
        Video        |   ✅  |   ❌   |  ❌
        Action       |   ✅  |   ✅   |  ✅
        VLM          |   ❌  |   ❌   |  ✅
    """

    def __init__(self, config: MotusWanVlmDirectMaskConfig):
        super().__init__()
        self.config = config

        # Set unified data type for the model
        self.dtype = torch.bfloat16

        # Initialize video/VLM backbones
        load_backbones = True if config.load_pretrained_backbones is None else bool(config.load_pretrained_backbones)

        # Initialize video model (WAN 5B)
        logger.info("Initializing WAN 5B video model...")
        if load_backbones:
            self.video_model = WanVideoModel.from_pretrained(
                checkpoint_path=config.wan_checkpoint_path,
                vae_path=config.vae_path,
                config_path=config.wan_config_path,
                precision=config.video_precision
            )
        else:
            self.video_model = WanVideoModel.from_config(
                config_path=config.wan_config_path,
                vae_path=config.vae_path,
                device="cuda",
                precision=config.video_precision
            )

        # Initialize VLM (Qwen3-VL-2B, trainable)
        logger.info("Initializing VLM (Qwen3-VL-2B, trainable{})...".format(" - FROZEN" if config.vlm_frozen else ""))
        if load_backbones:
            self.vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.vlm_checkpoint_path,
                dtype=self.dtype,
                device_map="cuda",
                trust_remote_code=True
            )
            logger.info("Load pretrained VLM...")
        else:
            vlm_cfg = AutoConfig.from_pretrained(config.vlm_checkpoint_path, trust_remote_code=True)
            self.vlm_model = Qwen3VLForConditionalGeneration._from_config(vlm_cfg, torch_dtype=self.dtype)
            self.vlm_model.to(device="cuda", dtype=self.dtype)
            logger.info("Initializing VLM from config...")

        # Freeze VLM parameters if specified
        if config.vlm_frozen:
            for param in self.vlm_model.parameters():
                param.requires_grad = False
            logger.info("VLM parameters frozen")
        else:
            logger.info("VLM parameters are TRAINABLE")

        # Keep VLM complete (do not truncate)
        vlm_num_layers = len(self.vlm_model.model.language_model.layers)
        logger.info(f"VLM kept complete with {vlm_num_layers} layers")

        # Get WAN and VLM configurations directly
        wan_dim = getattr(self.video_model.wan_model.config, 'dim', 3072)
        wan_num_heads = getattr(self.video_model.wan_model.config, 'num_heads', 24)
        wan_head_dim = wan_dim // wan_num_heads

        vlm_dim = self.vlm_model.config.text_config.hidden_size
        vlm_num_heads = self.vlm_model.config.text_config.num_attention_heads
        vlm_num_kv_heads = getattr(self.vlm_model.config.text_config if hasattr(self.vlm_model.config, 'text_config') else self.vlm_model.config, 'num_key_value_heads', vlm_num_heads)
        vlm_num_hidden_layers = self.vlm_model.config.text_config.num_hidden_layers
        vlm_head_dim = vlm_dim // vlm_num_heads

        logger.info(f"Model configurations:")
        logger.info(f"  WAN 5B: {wan_num_heads} heads × {wan_head_dim} head_dim = {wan_dim}D, {config.num_layers} layers")
        logger.info(f"  VLM 2B: {vlm_num_heads} Q heads, {vlm_num_kv_heads} KV heads × {vlm_head_dim} head_dim = {vlm_dim}D, {vlm_num_layers} layers")
        logger.info(f"  Layer mapping: VLM 0-27 -> WAN 0-27, VLM 26->WAN 28, VLM 27->WAN 29")

        # Create config dictionaries for ActionExpert
        wan_config = {
            'dim': wan_dim,
            'num_heads': wan_num_heads,
            'head_dim': wan_head_dim
        }

        # Initialize action expert with unified configs
        logger.info("Initializing Action Expert...")

        # Determine chunk_size based on training mode
        if config.training_mode == 'pretrain':
            action_chunk_size_for_expert = config.action_chunk_size
        else:
            action_chunk_size_for_expert = config.action_chunk_size + 1  # include state token

        # Configure registers by mode: no registers in pretrain, keep default (e.g., 4) in finetune
        num_registers = 0 if config.training_mode == 'pretrain' else 4

        action_config = ActionExpertConfig(
            dim=config.action_expert_dim,
            ffn_dim=config.action_expert_dim * config.action_expert_ffn_dim_multiplier,
            num_layers=config.num_layers,
            state_dim=config.action_state_dim,
            action_dim=config.action_dim,
            chunk_size=action_chunk_size_for_expert,
            num_registers=num_registers,
            video_feature_dim=wan_dim,
            causal=False,
            eps=config.action_expert_norm_eps,
            training_mode=config.training_mode,
        )

        if config.action_pretrained_path:
            logger.info(f"Loading Action Expert from pre-trained backbone: {config.action_pretrained_path}")
            self.action_expert = ActionExpert.from_pretrained(
                config=action_config,
                action_pretrained_path=config.action_pretrained_path,
                wan_config=wan_config,
                device='cpu',
                torch_dtype=self.dtype,
            )
        else:
            self.action_expert = ActionExpert(action_config, wan_config)

        # Move models to device - need device before initializing qwen3_module
        self.device = next(self.video_model.parameters()).device
        self.action_expert.to(device=self.device, dtype=self.dtype)

        # Initialize Qwen3-VL Module (per-layer QKV projections for direct MoT)
        logger.info("Initializing Qwen3-VL Module for direct MoT...")
        qwen3_config = Qwen3VLWanConfig(
            vlm_dim=config.vlm_dim,
            head_dim=config.qwen3_expert_head_dim,
            num_heads=config.qwen3_expert_num_heads,
            num_layers=config.qwen3_expert_num_layers,  # 30 layers (matches WAN)
            eps=config.qwen3_expert_norm_eps,
        )

        self.qwen3_module = Qwen3VLWanModule(self.vlm_model, qwen3_config, self.dtype, self.device)

        # Set time embedding layers to float32 for numerical stability
        self.action_expert.time_embedding.to(dtype=torch.float32)
        self.action_expert.time_projection.to(dtype=torch.float32)

        # Pre-compute grid_sizes for training batch size
        lat_T = 1 + config.num_video_frames // 4
        lat_H = config.video_height // 32
        lat_W = config.video_width // 32
        batch_size = config.batch_size
        self.grid_sizes = torch.tensor(
            [lat_T, lat_H, lat_W],
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0).expand(batch_size, -1)  # [batch_size, 3] - pre-expanded

        logger.info(f"Pre-computed grid_sizes: T={lat_T}, H={lat_H}, W={lat_W}")

        # Initialize modular components
        self.video_module = VideoModule(self.video_model, self.dtype, self.device, self.grid_sizes)
        self.action_module = ActionModule(self.action_expert, self.config, self.video_model, self.vlm_model, self.dtype, self.device)

        # Initialize t distributions from config
        time_dist_config = getattr(config, 'time_distribution', {})
        model_config = {
            'timestep_sample_method': time_dist_config.get('timestep_sample_method', 'logit_normal'),
            'sigmoid_scale': time_dist_config.get('sigmoid_scale', 1.0),
            'min_t': time_dist_config.get('min_t', 0.0),
            'max_t': time_dist_config.get('max_t', 1.0)
        }

        # Flow-Matching scheduler for training (video branch only)
        try:
            self.fm_train_scheduler = FlowMatchScheduler(
                shift=5.0,
                sigma_min=0.0,
                extra_one_step=True,
                num_train_timesteps=1000
            )
            # Enable training mode to build per-timestep weights (if used)
            self.fm_train_scheduler.set_timesteps(num_inference_steps=1000, training=True)
            logger.info("Initialized FlowMatchScheduler for training (video)")
        except Exception as e:
            logger.warning(f"Failed to init FlowMatchScheduler: {e}")

        # Flow-Matching scheduler for training (action branch)
        try:
            self.fm_train_scheduler_action = FlowMatchScheduler(
                shift=5.0,
                sigma_min=0.0,
                extra_one_step=True,
                num_train_timesteps=1000
            )
            # Enable training mode for action as well
            self.fm_train_scheduler_action.set_timesteps(num_inference_steps=1000, training=True)
            logger.info("Initialized FlowMatchScheduler for training (action)")
        except Exception as e:
            logger.warning(f"Failed to init FlowMatchScheduler for action: {e}")

        # Log parameter counts
        self.log_parameter_counts()

    def log_parameter_counts(self):
        """Log detailed parameter counts for each component."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        video_params = sum(p.numel() for p in self.video_model.parameters())
        action_params = sum(p.numel() for p in self.action_expert.parameters())
        vlm_params = sum(p.numel() for p in self.vlm_model.parameters())
        qwen3_module_params = sum(p.numel() for p in self.qwen3_module.parameters())

        vlm_trainable_params = sum(p.numel() for p in self.vlm_model.parameters() if p.requires_grad)
        qwen3_module_trainable_params = sum(p.numel() for p in self.qwen3_module.parameters() if p.requires_grad)

        logger.info(f"MotusWanVlmDirect parameter breakdown:")
        logger.info(f"  Total parameters: {total_params / 1e9:.2f}B")
        logger.info(f"  Trainable parameters: {trainable_params / 1e9:.2f}B")
        logger.info(f"  Video Model (WAN 5B): {video_params / 1e9:.2f}B")
        logger.info(f"  Action Expert: {action_params / 1e6:.1f}M")
        logger.info(f"  VLM (Qwen3-VL-2B): {vlm_params / 1e9:.2f}B (trainable: {vlm_trainable_params / 1e9:.2f}B)")
        logger.info(f"  Qwen3 Module (projections): {qwen3_module_params / 1e6:.1f}M")

    def load_checkpoint(self, path: str, strict: bool = True) -> Dict:
        """Load model checkpoint."""
        # Handle directory path
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            checkpoint_file = checkpoint_path / "mp_rank_00_model_states.pt"
            if not checkpoint_file.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
            path = str(checkpoint_file)

        # Load state dict
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['module']
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
        logger.info(f"Checkpoint loaded from {path}: missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")

        # Return additional state
        additional_state = {k: v for k, v in checkpoint.items()
                          if k not in ['module', 'config']}
        return additional_state

    def load_pretrain_weights(self, path: str) -> None:
        """Load weights from a pretrain checkpoint when current mode is finetune.

        Skips layers that depend on state vs action-only differences:
          - action_expert.input_encoder.*
          - action_expert.decoder.*
        """
        if self.config.training_mode != 'finetune':
            raise ValueError("load_pretrain_weights should be called only in finetune mode")

        # Handle directory path - try two possible locations
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            # Try two possible paths
            possible_paths = [
                checkpoint_path / "pytorch_model" / "mp_rank_00_model_states.pt",
                checkpoint_path / "mp_rank_00_model_states.pt",
            ]

            checkpoint_file = None
            for p in possible_paths:
                if p.exists():
                    checkpoint_file = p
                    logger.info(f"Found checkpoint: {checkpoint_file}")
                    break

            if checkpoint_file is None:
                raise FileNotFoundError(
                    f"Checkpoint not found. Tried:\n"
                    f"  - {possible_paths[0]}\n"
                    f"  - {possible_paths[1]}"
                )
            path = str(checkpoint_file)
        else:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        logger.info(f"Loading pretrain weights from {path}")
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('module', checkpoint)

        filtered = {}
        for k, v in state_dict.items():
            if ('action_expert.input_encoder' in k or 'action_expert.decoder' in k):
                continue
            filtered[k] = v

        missing, unexpected = self.load_state_dict(filtered, strict=False)
        logger.info(f"Loaded pretrain weights (filtered). Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    def training_step(
        self,
        first_frame: torch.Tensor,         # [B, C, H, W] - first frame
        video_frames: torch.Tensor,       # [B, num_frames, C, H, W] - target frames
        state: torch.Tensor = None,       # [B, state_dim] - robot state
        actions: torch.Tensor = None,     # [B, chunk_size, action_dim] - actions
        language_embeddings: Optional[List[torch.Tensor]] = None,  # Pre-encoded T5 embeddings for WAN
        vlm_inputs: Optional[List] = None,  # Complete VLM inputs from dataset
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        UniDiffuser training step with three modalities (WAN + Action + VLM direct).

        Args:
            first_frame: First video frame for Teacher Forcing
            video_frames: Target video frames
            state: Initial robot state
            actions: Target action sequence
            language_embeddings: Pre-encoded T5 embeddings for WAN model
            vlm_inputs: Complete VLM inputs from dataset
            return_dict: Whether to return detailed outputs

        Returns:
            Dictionary containing losses and metrics
        """
        import time  # Import for profiling
        # Reset timing for this step
        if not hasattr(self, '_timing'):
            self._timing = {}
        else:
            self._timing = {}

        B = video_frames.shape[0]

        # 1. Video pipeline
        # Normalize/format
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)  # [B, C, 1, H, W]
        video_normalized = (video_frames * 2.0 - 1.0).permute(0, 2, 1, 3, 4)  # [B, C, num_frames, H, W]
        full_video = torch.cat([first_frame_norm, video_normalized], dim=2)  # [B, C, frames+1, H, W]

        # Encode video using VAE
        t_vae_start = time.time()
        with torch.no_grad():
            clean_full_latent = self.video_model.encode_video(full_video.to(self.dtype))  # [B, 48, latent_frames, H', W']
            condition_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))  # [B, 48, 1, H', W']
        self._timing['vae_encode'] = time.time() - t_vae_start

        # Flow-Matching noise mixture
        timestep_id = torch.randint(0, self.fm_train_scheduler.num_train_timesteps, (B,))
        # Scalar timesteps (0..num_train_timesteps) for time embedding
        video_t_embed = self.fm_train_scheduler.timesteps[timestep_id].to(dtype=self.dtype, device=self.device)  # [B]
        # Sigma for noise mixture
        sigma = self.fm_train_scheduler.sigmas[timestep_id].to(dtype=self.dtype, device=self.device).view(B, 1, 1, 1, 1)
        video_noise = torch.randn_like(clean_full_latent, dtype=self.dtype)
        noisy_video_latent = clean_full_latent * (1 - sigma) + video_noise * sigma
        # Teacher Forcing on the first frame
        noisy_video_latent[:, :, 0:1] = condition_frame_latent
        # Flow-Matching target: noise - clean
        video_target = video_noise - clean_full_latent
        video_target[:, :, 0:1] = 0

        # Latent to Tokens
        video_tokens = self.video_module.prepare_input(noisy_video_latent.to(self.dtype))

        # 2. Action pipeline
        timestep_id_action = torch.randint(0, self.fm_train_scheduler_action.num_train_timesteps, (B,))
        # Discrete timesteps for time embedding (0..num_train_timesteps)
        action_t_embed = self.fm_train_scheduler_action.timesteps[timestep_id_action].to(dtype=self.dtype, device=self.device)  # [B]
        # Sigma for action noise mixture
        sigma_action = self.fm_train_scheduler_action.sigmas[timestep_id_action].to(dtype=self.dtype, device=self.device).view(B, 1, 1)
        action_noise = torch.randn_like(actions, dtype=self.dtype)
        noisy_actions = actions * (1 - sigma_action) + action_noise * sigma_action
        action_target = action_noise - actions

        # Encode Action Chunk with optional Registers
        if self.action_expert.config.num_registers > 0 and self.action_expert.registers is not None:
            registers = self.action_expert.registers.expand(B, -1, -1)  # [B, num_registers, dim]
        else:
            registers = None
        if self.config.training_mode == 'pretrain':
            action_tokens = self.action_expert.input_encoder(None, noisy_actions, registers)
        else:
            state_tokens = state.unsqueeze(1).to(self.dtype)
            action_tokens = self.action_expert.input_encoder(state_tokens, noisy_actions, registers)

        # Extract VLM per-layer features (28 VLM layers -> 30 layers output)
        t_vlm_start = time.time()
        vlm_per_layer = self.qwen3_module.extract_per_layer_features(vlm_inputs)  # List of 30 tensors
        self._timing['vlm_extract'] = time.time() - t_vlm_start
        assert len(vlm_per_layer) == 30, f"Expected 30 VLM layers, got {len(vlm_per_layer)}"

        # Time embeddings
        video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(video_t_embed, video_tokens.shape[1])
        action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(action_t_embed, action_tokens.shape[1])

        # T5 preprocess
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)

        # 3. MoT forward
        t_mot_start = time.time()
        # Initialize component timing accumulators
        wan_attn_time = 0
        wan_ffn_time = 0
        action_attn_time = 0
        action_ffn_time = 0
        vlm_attn_time = 0
        vlm_ffn_time = 0

        with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
            # Process through 30 layers - modality-grouped execution
            # Initialize vlm_tokens with first layer VLM features

            for layer_idx in range(self.config.num_layers):
                # VLM tokens for this layer: base layer features + accumulated MoT updates
                # Similar to original Motus where und_tokens is updated layer by layer
                vlm_base = vlm_per_layer[layer_idx]  # Base VLM features for this layer
                if layer_idx == 0:
                    vlm_tokens = vlm_base
                else:
                    # Add the difference between current and previous layer, plus accumulated updates
                    # This allows vlm_tokens to evolve through layers like video/action tokens
                    vlm_tokens = (vlm_base + vlm_tokens)/2

                # Compute AdaLN modulation once per layer using pre-computed parameters
                video_adaln_modulation = self.video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
                action_adaln_modulation = self.action_module.compute_adaln_modulation(action_adaln_params, layer_idx)

                # Trimodal MoT: WAN + Action + VLM joint attention
                t_joint_attn = time.time()
                video_tokens, action_tokens, vlm_tokens = self.video_module.process_joint_attention(
                    video_tokens, action_tokens, vlm_tokens,
                    video_adaln_modulation, action_adaln_modulation, layer_idx,
                    self.action_expert.blocks[layer_idx],
                    self.qwen3_module.blocks[layer_idx]
                )
                joint_attn_time = time.time() - t_joint_attn

                # Split joint attention time proportionally
                wan_attn_time += joint_attn_time / 3
                action_attn_time += joint_attn_time / 3
                vlm_attn_time += joint_attn_time / 3

                # WAN cross
                t_cross_attn = time.time()
                video_tokens = self.video_module.process_cross_attention(video_tokens, video_adaln_params, layer_idx, processed_t5_context)
                wan_attn_time += time.time() - t_cross_attn

                # FFNs: WAN, Action (VLM FFN is internal, already computed via extract_per_layer_features)
                t_ffn = time.time()
                video_tokens = self.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
                wan_ffn_time += time.time() - t_ffn

                t_ffn = time.time()
                action_tokens = self.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
                action_ffn_time += time.time() - t_ffn

                # VLM has no FFN (internal FFN already computed in extract_per_layer_features)
                # Just return vlm_tokens unchanged
                vlm_tokens = self.qwen3_module.process_ffn(vlm_tokens, layer_idx)

            # 4. Heads + Losses
            t_output_start = time.time()
            video_pred = self.video_module.apply_output_head(video_tokens, video_head_time_emb)
            action_pred_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
            up_len = action_pred_full.shape[1] - self.action_expert.config.num_registers
            # Slice predicted actions depending on mode
            if self.config.training_mode == 'pretrain':
                action_pred = action_pred_full[:, :up_len, :]
            else:
                action_pred = action_pred_full[:, 1:up_len, :]
            self._timing['output_heads'] = time.time() - t_output_start

        # Store component timings
        self._timing['wan_attn'] = wan_attn_time
        self._timing['wan_ffn'] = wan_ffn_time
        self._timing['action_attn'] = action_attn_time
        self._timing['action_ffn'] = action_ffn_time
        self._timing['vlm_attn'] = vlm_attn_time
        self._timing['vlm_ffn'] = vlm_ffn_time
        self._timing['mot_layers'] = time.time() - t_mot_start

        # Video loss (mask the first frame)
        video_pred_masked = video_pred.clone()
        video_pred_masked[:, :, 0:1] = 0
        video_loss = torch.nn.functional.mse_loss(video_pred_masked, video_target, reduction='mean')

        # Action loss
        action_loss = torch.nn.functional.mse_loss(action_pred, action_target, reduction='mean')

        total_loss = (
            self.config.video_loss_weight * video_loss +
            self.config.action_loss_weight * action_loss
        )

        if return_dict:
            return {
                'total_loss': total_loss,
                'video_loss': video_loss,
                'action_loss': action_loss,
                'video_timestep_mean': sigma.float().mean().item(),
                'action_timestep_mean': sigma_action.float().mean().item(),
            }

    def inference_step(
        self,
        first_frame: torch.Tensor,
        state: torch.Tensor = None,
        num_inference_steps: int = 50,
        language_embeddings: Optional[List[torch.Tensor]] = None,
        vlm_inputs: Optional[List] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Joint inference for video and action prediction.

        Args:
            first_frame: Initial frame [B, C, H, W]
            state: Initial robot state [B, state_dim]
            num_inference_steps: Number of denoising steps
            language_embeddings: Pre-encoded T5 embeddings for WAN model
            vlm_inputs: VLM inputs

        Returns:
            Tuple of (predicted_frames, predicted_actions)
        """
        B = first_frame.shape[0]

        language_embeddings = [emb.to(self.device).to(self.dtype) for emb in language_embeddings]
        state = state.to(self.device).to(self.dtype)
        first_frame = first_frame.to(self.device).to(self.dtype)

        # 1. Video/Action latents init
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)   # [0,1] -> [-1,1], [B, C, 1, H, W]
        with torch.no_grad():
            condition_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))   # [B, C', 1, H', W']

        # Init video/action latents
        B, C_latent, f_latent, H_latent, W_latent = condition_frame_latent.shape
        num_total_latent_frames = 1 + self.config.num_video_frames // 4
        video_latent = torch.randn((B, C_latent, num_total_latent_frames, H_latent, W_latent), device=self.device, dtype=self.dtype)
        video_latent[:, :, 0:1] = condition_frame_latent
        action_shape = (B, self.config.action_chunk_size, self.config.action_dim)
        action_latent = torch.randn(action_shape, device=self.device, dtype=self.dtype)

        # 2. VLM per-layer features and T5 context
        vlm_per_layer = self.qwen3_module.extract_per_layer_features(vlm_inputs)
        assert len(vlm_per_layer) == 30, f"Expected 30 VLM layers, got {len(vlm_per_layer)}"

        # T5 preprocess
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)

        # 3. Denoising loop: from noise (t=1) to clean (t=0)
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=self.device, dtype=self.dtype)
        for i in range(num_inference_steps):
            # Timesteps
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t
            video_t_scaled = (t * 1000).expand(B).to(self.dtype)
            action_t_scaled = (t * 1000).expand(B).to(self.dtype)

            # Tokens with Registers
            video_tokens = self.video_module.prepare_input(video_latent.to(self.dtype))
            state_tokens = state.unsqueeze(1).to(self.dtype)
            registers = self.action_expert.registers.expand(B, -1, -1)  # [B, num_registers, dim]
            action_tokens = self.action_expert.input_encoder(state_tokens, action_latent, registers)

            # Extract VLM per-layer features for this step (VLM is updated each step)
            vlm_per_layer = self.qwen3_module.extract_per_layer_features(vlm_inputs)

            # Trimodal MoT forward - joint denoising for WAN, Action, VLM
            with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
                # Time embeddings
                video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(video_t_scaled, video_tokens.shape[1])
                action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(action_t_scaled, action_tokens.shape[1])

                # Process through all layers - trimodal denoising of WAN, Action, VLM
                # Initialize vlm_tokens with first layer VLM features
                vlm_tokens = vlm_per_layer[0]  # [B, T_vlm, 2048]

                for layer_idx in range(self.config.num_layers):
                    # VLM tokens for this layer: base layer features + accumulated MoT updates
                    # Similar to training_step and original Motus where und_tokens evolves through layers
                    vlm_base = vlm_per_layer[layer_idx]  # Base VLM features for this layer
                    if layer_idx == 0:
                        vlm_tokens = vlm_base
                    else:
                        # Mix VLM layer output with accumulated updates (average)
                        vlm_tokens = (vlm_base + vlm_tokens) / 2

                    # Compute AdaLN modulation using pre-computed parameters
                    video_adaln_modulation = self.video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
                    action_adaln_modulation = self.action_module.compute_adaln_modulation(action_adaln_params, layer_idx)

                    # Trimodal joint attention: WAN + Action + VLM
                    video_tokens, action_tokens, vlm_tokens = self.video_module.process_joint_attention(
                        video_tokens, action_tokens, vlm_tokens,
                        video_adaln_modulation, action_adaln_modulation, layer_idx,
                        self.action_expert.blocks[layer_idx],
                        self.qwen3_module.blocks[layer_idx]
                    )

                    # WAN cross-attention with T5 embeddings
                    video_tokens = self.video_module.process_cross_attention(
                        video_tokens, video_adaln_params, layer_idx, processed_t5_context
                    )

                    # FFNs: WAN, Action (VLM FFN is internal)
                    video_tokens = self.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
                    action_tokens = self.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
                    vlm_tokens = self.qwen3_module.process_ffn(vlm_tokens, layer_idx)

                # Heads (velocities)
                video_velocity = self.video_module.apply_output_head(video_tokens, video_head_time_emb)
                action_pred_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
                # Extract middle action chunk (skip first state token and last register tokens)
                action_velocity = action_pred_full[:, 1:-self.action_expert.config.num_registers, :]

                # Euler integration
                video_latent = video_latent + video_velocity * dt
                action_latent = action_latent + action_velocity * dt

                # Teacher Forcing
                video_latent[:, :, 0:1] = condition_frame_latent

        # 4. Decode outputs
        with torch.no_grad():
            decoded_frames = self.video_model.decode_video(video_latent)
            predicted_frames = decoded_frames[:, :, 1:]  # Skip first frame (condition)
            predicted_frames = (predicted_frames + 1.0) / 2.0  # [-1,1] to [0,1]
            predicted_frames = torch.clamp(predicted_frames, 0, 1).float()

        predicted_actions = action_latent.float()  # [B, action_chunk_size, 14]

        return predicted_frames, predicted_actions


def test_motus_wan_vlm_direct():
    """Test the complete model."""
    print("Testing MotusWanVlmDirect...")

    config = MotusWanVlmDirectConfig()

    try:
        model = MotusWanVlmDirect(config)
        print("Model created successfully")

        # Test parameter counting
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params / 1e9:.2f}B")

    except Exception as e:
        print(f"Model creation failed: {e}")
        print("This is expected without actual pretrained weights")


if __name__ == "__main__":
    test_motus_wan_vlm_direct()
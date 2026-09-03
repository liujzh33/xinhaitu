# Understanding Expert Model (No VLM version)
# Removed VLM adapter dependencies - uses learnable queries instead

import torch
import torch.nn as nn
import logging
from dataclasses import dataclass
import sys
from pathlib import Path

# Import WAN's components for consistency
project_root = Path(__file__).parent.parent
bak_root = project_root / "bak"
if str(bak_root.resolve()) not in sys.path:
    sys.path.insert(0, str(bak_root.resolve()))

from wan.modules.model import WanRMSNorm, WanLayerNorm

logger = logging.getLogger(__name__)


@dataclass
class UndExpertConfig:
    """Configuration for Understanding Expert model (No VLM version)."""
    # Architecture
    dim: int = 512                   # Hidden dimension for understanding expert
    ffn_dim: int = 2048              # FFN dimension (computed from dim * multiplier)
    num_layers: int = 30             # Number of layers (unified with WAN and Action)

    # Training
    eps: float = 1e-5                # Layer norm epsilon


class UndExpertBlock(nn.Module):
    """
    Understanding Expert Block - almost identical to ActionExpertBlock.
    
    Only provides projections for trimodal joint attention with WAN, no registers.
    """
    
    def __init__(self, config: UndExpertConfig, wan_config: dict):
        super().__init__()
        self.config = config
        
        # Layer norms (WAN style) - only need one for joint attention and one for FFN
        self.norm1 = WanLayerNorm(config.dim, eps=config.eps)  # For trimodal joint attention
        self.norm2 = WanLayerNorm(config.dim, eps=config.eps)  # For FFN
        
        # WAN-side understanding projections and norms (MoT: understanding -> WAN head space for trimodal joint attention)
        self.wan_num_heads = wan_config['num_heads']
        self.wan_head_dim = wan_config['head_dim']
        self.wan_dim = wan_config['dim']
        assert self.wan_num_heads * self.wan_head_dim == self.wan_dim
        self.wan_und_qkv = nn.Parameter(
            torch.randn(3, self.wan_num_heads, config.dim, self.wan_head_dim)
            / (config.dim * self.wan_head_dim) ** 0.5
        )
        self.wan_und_o = nn.Linear(self.wan_dim, config.dim, bias=False)
        # normalize Q/K in WAN unified dim
        self.wan_und_norm_q = WanRMSNorm(self.wan_dim, eps=config.eps)
        self.wan_und_norm_k = WanRMSNorm(self.wan_dim, eps=config.eps)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.dim, config.ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(config.ffn_dim, config.dim)
        )


class UndExpert(nn.Module):
    """
    Understanding Expert model (No VLM version).

    Key features:
    - No VLM adapter (uses learnable queries from UndModule)
    - No registers
    - Configurable FFN ratio
    - No decoder
    """

    def __init__(self, config: UndExpertConfig, wan_config: dict = None, vlm_config: dict = None):
        super().__init__()
        self.config = config
        self.freq_dim = 256  # Sinusoidal embedding dimension

        # Transformer blocks (same number as WAN/Action for 1:1 correspondence)
        if wan_config is not None:
            self.blocks = nn.ModuleList([
                UndExpertBlock(config, wan_config) for _ in range(config.num_layers)
            ])
        else:
            # Fallback: create blocks with default WAN config (for backward compatibility)
            self.blocks = nn.ModuleList([
                UndExpertBlock(config, {'dim': 3072, 'num_heads': 24, 'head_dim': 128})
                for _ in range(config.num_layers)
            ])

# Models package for Motus

# Original Motus models
from .motus import Motus, MotusConfig
from .wan_model import WanVideoModel
from .action_expert import ActionExpert, ActionExpertConfig
from .und_expert import UndExpert, UndExpertConfig

# New WAN-based models with VLM direct MoT
from .motus_wan_vlm_direct import MotusWanVlmDirect, MotusWanVlmDirectConfig
from .qwen3_module_wan import Qwen3VLWanModule, Qwen3VLWanConfig

__all__ = [
    # Original models
    "Motus", "MotusConfig",
    "WanVideoModel",
    "ActionExpert", "ActionExpertConfig",
    "UndExpert", "UndExpertConfig",
    # WAN-based models with VLM direct MoT
    "MotusWanVlmDirect", "MotusWanVlmDirectConfig",
    "Qwen3VLWanModule", "Qwen3VLWanConfig",
]

import os
import random
import numpy as np
import builtins
import fcntl
import glob
import functools
from typing import Union

import huggingface_hub

import picotron.process_group_manager as pgm
import torch, torch.distributed as dist

# -----------------------------------------------------------------------------
# Global Device Management for Apple Silicon (MPS/CPU)
# -----------------------------------------------------------------------------

_global_device: torch.device = "mps" if torch.backends.mps.is_available() else "cpu"

def set_global_device(device: Union[torch.device, str]) -> None:
    """Set the global device for all operations.
    
    Args:
        device: Either a torch.device or a string ('mps', 'cpu').
                Defaults to MPS if available, otherwise CPU.
    """
    global _global_device
    if isinstance(device, str):
        device = torch.device(device)
    _global_device = device

def get_global_device() -> torch.device:
    """Get the global device. Auto-initializes to MPS if available, else CPU."""
    global _global_device
    if _global_device is None:
        if torch.backends.mps.is_available():
            _global_device = torch.device('mps')
        else:
            _global_device = torch.device('cpu')
    return _global_device

def with_device(func=None, *, override=None):
    """Decorator that injects a device into function kwargs if not provided.
    
    The decorated function must accept a 'device' keyword argument.
    If 'device' is not passed by the caller, it will be set to:
    - The override device if specified (e.g., 'cpu' for distributed ops)
    - Otherwise, get_global_device()
    
    Examples:
        @with_device
        def train_step(model, data_loader, device):
            input_ids = batch["input_ids"].to(device)
            ...
        
        @with_device(override='cpu')  # For distributed ops that don't support MPS
        def average_loss(loss, device):
            reduced = torch.tensor([loss], device=device)
            dist.all_reduce(reduced)
            ...
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if 'device' not in kwargs:
                if override is not None:
                    kwargs['device'] = torch.device(override)
                else:
                    kwargs['device'] = get_global_device()
            return fn(*args, **kwargs)
        return wrapper
    
    # Handle both @with_device and @with_device(override='cpu') syntax
    if func is not None:
        return decorator(func)
    return decorator

@with_device
def get_memory_usage_gb(device: torch.device = None) -> float:
    """Get memory usage in GB for the given device (MPS/CPU)."""
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1e9
    return 0.0  # CPU doesn't have a direct memory API

def print(*args, is_print_rank=True, **kwargs):
    """ solves multi-process interleaved print problem """
    if not is_print_rank: return
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            builtins.print(*args, **kwargs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

def set_all_seed(seed):
    """Set random seeds for reproducibility across all backends."""
    for module in [random, np.random]: module.seed(seed)
    torch.manual_seed(seed)
    
def to_readable_format(num, precision=2):
    if num >= 1e12:
        return f"{num / 1e12:.{precision}f}T"
    elif num >= 1e9:
        return f"{num / 1e9:.{precision}f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.{precision}f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"

import subprocess

def get_apple_silicon_flops():
    try:
        command = ["sysctl", "-n", "machdep.cpu.brand_string"]
        output = subprocess.check_output(command).decode().strip()
        
        # Approximate FP32 TFLOPS based on GPU benchmarks
        flops_map = {
            "M1": 2.6e12,
            "M1 Pro": 5.2e12,
            "M1 Max": 10.4e12,
            "M1 Ultra": 21e12,
            "M2": 3.6e12,
            "M2 Pro": 6.8e12,
            "M2 Max": 13.6e12,
            "M2 Ultra": 27e12,
            "M3": 4.3e12,
            "M3 Pro": 7.3e12,
            "M3 Max": 14e12,
            "M3 Ultra": 28e12,
        }

        # Sort keys by length descending to ensure "M1 Max" matches before "M1"
        sorted_keys = sorted(flops_map.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if key in output:
                return flops_map[key]
        
        raise ValueError(f"Unknown Apple Silicon: {output}")
    except Exception as e:
        raise ValueError(f"Error getting Apple Silicon FLOPS: {e}")

@with_device
def get_theoretical_flops(device: torch.device = None) -> float:
    """Get theoretical FLOPS for the device. Optimized for Apple Silicon."""
    if device.type == "mps":
        return get_apple_silicon_flops()
    # Fallback for CPU (conservative estimate)
    return 100e9  # 100 GFLOPS as a baseline for CPU

def get_mfu(tokens_per_second, num_params, model_config, theoretical_flops=None):
    if theoretical_flops is None:
        theoretical_flops = 989.5 * 10 ** 12
    
    num_layers = model_config.num_hidden_layers
    hidden_dim = model_config.hidden_size
    seq_len = model_config.max_position_embeddings
    flops_per_token = 6 * num_params + 12 * num_layers * hidden_dim * seq_len
    mfu = tokens_per_second * flops_per_token / theoretical_flops * 100 # percentage
    return mfu

def get_num_params(model):
    """Calculate total number of parameters accounting for tensor parallelism and pipeline parallelism.
    
    For TP: Parameters in attention/mlp/embed/final_proj are sharded, so multiply by tp_world_size
    For PP: Need to gather parameter counts across pipeline stages
    For DP: Parameters are replicated, so only count once
    
    Note: 
    FSDP: Parameters are sharded across data parallel ranks
    """
    tp_world_size = pgm.process_group_manager.tp_world_size
    
    # Count parameters in current PP rank
    local_num_params = 0
    for name, param in model.named_parameters():
        # Parameters split across TP ranks
        # TODO: LayerNorm is also split across TP ranks for sequence parallelism
        if any(tp_keyword in name.lower() for tp_keyword in ['attention', 'mlp', 'embed', 'final_proj']):
            local_num_params += param.numel() * tp_world_size
        else:
            # Parameters replicated across TP ranks (layer norm, biases)
            local_num_params += param.numel()
            
    # Use CPU for distributed ops (MPS doesn't support collective ops)
    param_counts = torch.tensor(local_num_params, device='cpu')
    
    # Sum up parameters across all PP ranks
    dist.all_reduce(param_counts, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.pp_group)
    
    return param_counts.item()
    
def assert_no_meta_tensors(model):
    meta_tensors = []
    for name, param in model.named_parameters():
        if param.device == torch.device("meta"):
            meta_tensors.append(f"Parameter '{name}' with shape {param.shape}")
    
    for name, buffer in model.named_buffers():
        if buffer.device == torch.device("meta"):
            meta_tensors.append(f"Buffer '{name}' with shape {buffer.shape}")
    
    assert len(meta_tensors) == 0, f"Found {len(meta_tensors)} meta tensors:\n" + "\n".join(meta_tensors)

@with_device(override='cpu')  # Distributed ops don't support MPS
def average_loss_across_dp_cp_ranks(loss, device: torch.device = None):
    """Average loss across data parallel and context parallel ranks."""
    reduced_loss = torch.tensor([loss if loss is not None else 0.0], dtype=torch.float32, device=device)
    if pgm.process_group_manager.pp_is_last_stage:
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.cp_dp_group)
        reduced_loss /= pgm.process_group_manager.cp_dp_world_size
    return reduced_loss.item()

def download_model(model_name, hf_token):
    dst = os.path.join("hf_model", model_name)
    os.makedirs(dst, exist_ok=True)
    # check if model is already downloaded
    if os.path.exists(os.path.join(dst, "config.json")):
        print(f"Model {model_name} already exists at {dst}")
        return
    # Download HF model safetensors at the "dst" directory
    huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
    print("Downloading SafeTensors files...")
    huggingface_hub.snapshot_download(model_name, repo_type="model", local_dir="hf_model_safetensors", token=hf_token,
                                      allow_patterns=["*.safetensors", "*.json"])
    # Check if the model has SafeTensors files
    if not glob.glob("hf_model_safetensors/*.safetensors"):
        raise ValueError(f"Model {model_name} does not have SafeTensors files.")
    print("SafeTensors files downloaded successfully! ✅")

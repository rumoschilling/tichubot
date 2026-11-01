import os
import torch

# Try to import pynvml for GPU utilization metrics
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except (ImportError, Exception):
    NVML_AVAILABLE = False

def get_system_stats(device: torch.device) -> dict[str, str]:
    stats = {}
    
    # CPU Load
    try:
        load1, load5, load15 = os.getloadavg()
        stats["cpu_load_1m"] = f"{load1:.2f}"
    except (OSError, AttributeError):
        stats["cpu_load_1m"] = "0.00"
        
    # RAM Available
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    stats["ram_avail_gb"] = f"{kb / (1024*1024):.1f}"
                    break
    except (OSError, IndexError, ValueError):
        stats["ram_avail_gb"] = "0.0"

    # GPU VRAM and Load
    if device.type == "cuda":
        try:
            idx = device.index if device.index is not None else 0
            
            allocated = torch.cuda.memory_allocated(device) / (1024**2)
            reserved = torch.cuda.memory_reserved(device) / (1024**2)
            total = torch.cuda.get_device_properties(device).total_memory / (1024**2)
            
            stats["vram_used_mb"] = f"{reserved:.0f}"
            stats["vram_total_mb"] = f"{total:.0f}"
            stats["vram_alloc_mb"] = f"{allocated:.0f}"
            
            if NVML_AVAILABLE:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                stats["gpu_util_pct"] = str(util.gpu)
            else:
                stats["gpu_util_pct"] = "0"
                
        except Exception:
            stats["vram_used_mb"] = "0"
            stats["vram_total_mb"] = "0"
            stats["gpu_util_pct"] = "0"
    else:
        stats["vram_used_mb"] = "0"
        stats["vram_total_mb"] = "0"
        stats["gpu_util_pct"] = "0"
        
    return stats

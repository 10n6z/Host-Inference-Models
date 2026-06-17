import os


def visible_cuda_devices() -> list[str]:
    raw = os.getenv("CUDA_VISIBLE_DEVICES") or os.getenv("NVIDIA_VISIBLE_DEVICES") or ""
    if raw.strip().lower() in ("", "all"):
        try:
            import torch
            if torch.cuda.is_available():
                return [str(i) for i in range(torch.cuda.device_count())]
        except Exception:
            return []
    return [part.strip() for part in raw.split(",") if part.strip() and part.strip().lower() not in ("none", "void")]


def has_multiple_cuda_devices() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.device_count() > 1
    except Exception:
        return False


def maybe_enable_multi_gpu(pipe):
    if not has_multiple_cuda_devices():
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        return pipe
    if hasattr(pipe, "reset_device_map"):
        pipe.reset_device_map()
    return pipe

"""
Единый выбор устройства (cuda → mps → cpu) и подавление шумных
предупреждений, которые возникают на MPS.
"""
import os
import warnings

import torch


def get_torch_device() -> str:
    """Возвращает лучшее доступное устройство: cuda:0 → mps → cpu."""
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def get_torch_dtype(device: str):
    """Подбирает подходящий dtype: bf16 на CUDA, fp16 на MPS, fp32 на CPU."""
    if device.startswith("cuda"):
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def easyocr_uses_gpu(device: str) -> bool:
    """EasyOCR официально поддерживает только CUDA."""
    return device.startswith("cuda")


def setup_environment() -> None:
    """Подавляет шумные warnings (pin_memory на MPS и т.п.), включает MPS-fallback."""
    # MPS: DataLoader пишет warning, что pin_memory не поддерживается.
    warnings.filterwarnings(
        "ignore",
        message=r".*'pin_memory' argument is set as true but not supported on MPS.*",
    )
    warnings.filterwarnings("ignore", message=r".*pin_memory.*MPS.*")
    # Некоторые операторы Qwen/VL ещё не реализованы на MPS — пусть падают на CPU,
    # а не выкидывают исключение.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # Снижаем лишний шум HF.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

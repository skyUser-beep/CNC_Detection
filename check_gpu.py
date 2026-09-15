try:
    import torch
except Exception as exc:
    print("PyTorch import failed:", exc)
    raise SystemExit(1)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2))

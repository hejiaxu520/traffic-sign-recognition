import torch

print("=" * 50)
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU设备名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"当前使用的GPU编号: {torch.cuda.current_device()}")
else:
    print("❌ CUDA不可用！请检查前面的安装步骤")
print("=" * 50)
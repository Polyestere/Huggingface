import torch

print("PyTorch 버전:", torch.__version__)
print("PyTorch에서 사용하는 CUDA 버전:", torch.version.cuda)
print("CUDA 사용 가능 여부:", torch.cuda.is_available())

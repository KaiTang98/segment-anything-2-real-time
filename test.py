import torch
from torch.backends.cuda import sdp_kernel

print("PyTorch:", torch.__version__)
print("SDPA  supported kernels:", sdp_kernel.supported_kernels())
print("SDPA  preferred kernel :", sdp_kernel.preferred_kernel(dtype=torch.float16))
try:
    import xformers
    print("xFormers:", xformers.__version__)
except ImportError:
    print("xFormers not installed")
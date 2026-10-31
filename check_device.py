
import platform
import torch

print("PyTorch version:", torch.__version__)
print("Python architecture:", platform.machine())

print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Selected device:", device)

# Verify that a tensor operation works
x = torch.randn(100, 100, device=device)
y = x @ x

print("Tensor device:", y.device)
print("Matrix multiplication successful:", y.shape)

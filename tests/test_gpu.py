import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

x = torch.randn(4096, 4096, device=device)
y = torch.randn(4096, 4096, device=device)

z = x @ y

print("GPU:", torch.cuda.get_device_name(0))
print("Result shape:", z.shape)
print("Allocated VRAM:",
      round(torch.cuda.memory_allocated() / 1024**3, 2),
      "GB")
import torch
import time

SIZE_GB = 10
n = SIZE_GB * 1024**3 // 2  # BF16 = 2 bytes per element

buf_cpu = torch.empty(n, dtype=torch.bfloat16, pin_memory=True)
buf_gpu = torch.empty(n, dtype=torch.bfloat16, device="cuda")

# Warmup
buf_gpu.copy_(buf_cpu)
torch.cuda.synchronize()

RUNS = 5

# Host -> Device
start = time.perf_counter()
for _ in range(RUNS):
    buf_gpu.copy_(buf_cpu)
    torch.cuda.synchronize()
h2d = SIZE_GB * RUNS / (time.perf_counter() - start)

# Device -> Host
start = time.perf_counter()
for _ in range(RUNS):
    buf_cpu.copy_(buf_gpu)
    torch.cuda.synchronize()
d2h = SIZE_GB * RUNS / (time.perf_counter() - start)

print(f"Host -> Device: {h2d:.1f} GB/s")
print(f"Device -> Host: {d2h:.1f} GB/s")

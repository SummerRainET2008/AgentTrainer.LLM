import torch
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Capability: {torch.cuda.get_device_capability(0)}")

# Verify BF16 (Blackwell's native strength)
try:
    a = torch.randn(1024, 1024, dtype=torch.bfloat16).cuda()
    b = torch.randn(1024, 1024, dtype=torch.bfloat16).cuda()
    c = a @ b
    print("BF16 MatMul on Blackwell: Success!")
except Exception as e:
    print(f"Blackwell Test Failed: {e}")

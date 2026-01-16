import torch
ctx = torch.load("context.pth", map_location="cpu")
print(type(ctx))
if isinstance(ctx, dict):
    print("keys:")
    for k, v in ctx.items():
        if torch.is_tensor(v):
            print(f"  {k}: tensor {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k}: {type(v)}")
else:
    print(ctx)
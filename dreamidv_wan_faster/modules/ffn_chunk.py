import types
import torch

def decide_chunks_by_N(N: int) -> int:
    """
    自适应策略：你可以随时调
    """
    print(f"Decide chunk: N={N}")
    if N <= 2048:
        return 1        # 不 chunk
    elif N <= 4096:
        return 2
    elif N <= 8192:
        return 4
    else:
        return 8


def _chunked_call(module, x: torch.Tensor):
    # x: [B, N, C]
    if x.dim() != 3:
        return module._orig_forward(x)

    B, N, C = x.shape

    # 第一次 forward：根据 N 决定 chunks
    if not hasattr(module, "_ffn_chunks"):
        chunks = decide_chunks_by_N(N)
        module._ffn_chunks = chunks
        print(f"[FFN-Chunk] auto select chunks={chunks} for N={N}")

    print(f"Existing chunk: N={N}")

    chunks = module._ffn_chunks
    if chunks <= 1:
        return module._orig_forward(x)

    chunk_size = max(1, N // chunks)
    y = torch.empty_like(x)

    for i in range(chunks):
        s = i * chunk_size
        if s >= N:
            break
        e = N if i == chunks - 1 else min(N, (i + 1) * chunk_size)
        y[:, s:e, :] = module._orig_forward(x[:, s:e, :])

    return y


def patch_model_ffn_chunking(model, verbose=True):
    patched = 0
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise AttributeError("model has no attribute 'blocks'")

    for idx, block in enumerate(blocks):
        ffn = getattr(block, "ffn", None)
        if ffn is None:
            continue

        if not hasattr(ffn, "_orig_forward"):
            ffn._orig_forward = ffn.forward

        def wrapped_forward(self_ffn, x, *args, **kwargs):
            if args or kwargs:
                return self_ffn._orig_forward(x, *args, **kwargs)
            return _chunked_call(self_ffn, x)

        ffn.forward = types.MethodType(wrapped_forward, ffn)
        patched += 1

    if verbose:
        print(f"[FFN-Chunk] patched {patched} FFN blocks with auto-chunking")

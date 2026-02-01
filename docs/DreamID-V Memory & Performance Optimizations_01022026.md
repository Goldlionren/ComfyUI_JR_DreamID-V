# DreamID-V Memory & Performance Optimizations

> This document explains **how and why DreamID-V can now run higher resolutions and longer videos without OOM**, and provides **practical guidelines** for choosing resolution, frame count, token size (N), and FFN chunk settings.

---

## 1. Background: Why DreamID-V OOMs Easily

DreamID-V (Wan2.x-based) uses a **DiT / Transformer-style backbone**, not a UNet.

Key properties:

* Tokens scale with **spatial resolution × temporal length**
* Each Transformer block contains:

  * Self-Attention
  * Cross-Attention
  * **FFN (Feed-Forward Network)** ← *major VRAM peak source*
* Reference video is encoded by a **video VAE**, which can also cause large transient VRAM spikes

Before optimization, peak VRAM came from **three independent sources**:

1. FFN activations (`[B, N, hidden] → [B, N, ffn_dim]`)
2. Attention temporary tensors (QKV, softmax)
3. VAE encode of reference video (entire clip at once)

---

## 2. FFN Chunking (Primary Breakthrough)

### 2.1 What is FFN chunking?

Standard FFN:

```
[B, N, C] → Linear(C → ffn_dim) → GELU → Linear(ffn_dim → C)
```

This creates a **huge intermediate tensor** of shape:

```
[B, N, ffn_dim]
```

FFN chunking splits the token dimension:

```
[B, N, C]
→ split into K chunks along N
→ process each chunk independently
→ concat back
```

Result:

* **Peak VRAM ∝ N / chunks**
* Output is numerically identical (no approximation)

---

### 2.2 Implementation

All FFN blocks are patched at runtime:

* Replace `ffn.forward`
* Chunk along token dimension `N`
* Auto or fixed chunk size

Example log:

```
[FFN-Chunk] patched 30 FFN blocks with auto-chunking
[FFN-Chunk] auto select chunks=8 for N=11520
```

---

### 2.3 Why FFN chunking works so well

* FFN is **stateless per token**
* No cross-token dependency
* Perfectly safe to chunk
* No quality degradation observed

> In practice, FFN chunking alone reduces peak VRAM by **30–60%**.

---

## 3. Understanding N (Token Count)

### 3.1 Definition of N

For DreamID-V:

```
N = T × (H / stride_h) × (W / stride_w)
```

Where:

* `T` = number of frames processed in one chunk
* `H, W` = input resolution
* `stride_h, stride_w` = VAE / patch stride (typically 8×8 or 16×16)

Example (576×1024, stride=8, T=16):

```
H' = 576 / 8 = 72
W' = 1024 / 8 = 128
N = 16 × 72 × 128 = 147,456   (raw)
```

After internal packing / model specifics, effective N often shows as:

```
~11k – 28k (observed in logs)
```

---

### 3.2 Why N matters

* **FFN VRAM ∝ N × ffn_dim**
* Attention VRAM ∝ N² (softmax) but heavily optimized
* Large N = exponential risk of OOM

---

## 4. Recommended FFN Chunk Settings

### 4.1 Practical Chunk Formula (Rule of Thumb)

```
recommended_chunks = ceil(N / 4000)
```

Clamp to a reasonable range:

```
chunks = min(max(recommended_chunks, 2), 16)
```

---

### 4.2 Recommended Map

| Resolution | Frames (per chunk) | Typical N | Recommended chunks |
| ---------- | ------------------ | --------- | ------------------ |
| 512p       | 16                 | ~6k       | 2–4                |
| 720p       | 16                 | ~9k       | 4–6                |
| 1024p      | 16                 | ~11–14k   | 6–8                |
| 1280p      | 16                 | ~18–22k   | 8–12               |
| 1280p      | 41                 | ~25–28k   | 8–16               |

> **Empirical sweet spot:**
> `chunks = 8` works extremely well for most 1024–1280p cases.

---

## 5. VAE Encode Optimization (Hidden OOM Source)

### 5.1 Original problem

Originally:

```python
vae.encode([video_frames])  # entire video at once
```

Problems:

* Massive temporary tensors
* Cold-start allocator fragmentation
* OOM during `F.pad` or convolution layers

---

### 5.2 Fix: Temporal Micro-Batch VAE Encode

Reference video is now encoded as:

```
[C, T, H, W]
→ split along T
→ encode T_chunk frames at a time
→ concat latents
```

Features:

* Automatic fallback on OOM (16 → 8 → 4 → 2 → 1 frames)
* Latents default to CPU
* GPU memory freed immediately after each slice

Result:

* Cold-start OOM eliminated
* Peak VRAM during VAE encode reduced dramatically

---

### 5.3 Fix: Temporal Micro-Batch VAE Decode

Reference video is now encoded as:

```
        outs = []
        for i in range(iter_):
            self._conv_idx = [0]
            out_i = self.decoder(
                x[:, :, i:i + 1, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx
            )
            outs.append(out_i)
        # one concat at the end -> much lower peak VRAM and fragmentation
        out = torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]
        # help GC
        try:
            del outs
        except Exception:
            pass
```

Features:

* one concat at the end -> much lower peak VRAM and fragmentation

Result:

* Much lower peak VRAM to reduce chance of OOM

---

## 6. Warmup Pass (Formalized “1-Second Trick”)

### 6.1 Motivation

Historically:

> “Run 1 frame / 1 second first, then real run won’t OOM.”

Root cause:

* CUDA allocator cold state
* Kernel / Triton / cuDNN initialization
* Attention / FFN kernel autotuning

---

### 6.2 Formal Warmup Pass

Now implemented as an **explicit warmup pass**:

* Runs once before long-video chunks
* Uses:

  * Same pipeline
  * Same resolution
  * Same backend
* Uses:

  * Short duration (e.g. 1s)
  * Small step count (e.g. 4)
* Output is **discarded**
* **No `empty_cache()` after warmup**

Result:

* Allocator stays “hot”
* First real chunk no longer slow or unstable

---

## 7. Segment-Boundary Cache Cleanup (Not Everywhere!)

### Correct rule:

| Location                 | Clear cache? |
| ------------------------ | ------------ |
| Inside FFN / attention   | ❌            |
| During sampling steps    | ❌            |
| After VAE encode         | ❌            |
| **Between video chunks** | ✅ (once)     |

Implementation:

* Only one `_soft_empty_cache()` call
* Executed **after a chunk fully finishes**
* Preserves kernel / allocator warm state

---

## 8. Final Observed Results (Real Hardware)

Example: **1280p, 5 seconds video**

| Metric         | Before   | After     |
| -------------- | -------- | --------- |
| Cold start     | OOM      | OK        |
| Peak VRAM      | ~17GB+   | ~15.4GB   |
| Inference VRAM | ~99%     | ~8GB      |
| Runtime 576x1024 5s      | ~20+ min | ~8.5 min  |
| Runtime 720x1280 5s      | OOM | ~18.5 min  |
| Quality        | –        | Identical |

---

## 9. Summary: Why This Works

This optimization stack works because:

1. **FFN chunking** reduces the largest activation tensor
2. **VAE temporal chunking** removes cold-start spikes
3. **Warmup pass** stabilizes allocator & kernels
4. **Segment-boundary cache cleanup** prevents leaks without killing performance

Together, these changes turn DreamID-V from:

> *“barely runnable with luck”*

into:

> *“predictable, tunable, and scalable”*

---
# DreamID-V 显存与性能优化说明（FFN Chunk / VAE 分段 / Warmup）

> 本文档说明 **DreamID-V（基于 Wan2.x / DiT 架构）** 如何在显存有限的情况下运行 **更高分辨率、更长视频**，并系统解释：
>
> * 为什么原始版本极易 OOM
> * FFN chunk 是如何成为关键突破点
> * N（token 数）到底是什么、为什么重要
> * 如何根据分辨率 / 帧数选择 chunk
> * 为什么“先跑 1 秒再跑 5 秒”真的有效
> * VAE encode / decode 中隐藏的显存陷阱

---

## 1. 背景：为什么 DreamID-V 很容易 OOM？

DreamID-V 使用的是 **DiT / Transformer-style backbone**，而不是传统的 UNet。

其核心特征是：

* token 数 **随空间分辨率 × 时间长度线性增长**
* 每个 Transformer Block 包含：

  * Self-Attention
  * Cross-Attention
  * **FFN（Feed-Forward Network） ← 最大显存来源**
* 参考视频通过 **Video VAE** 编码，也会产生巨大的瞬时显存峰值

在未优化前，显存峰值主要来自 **三处**：

1. **FFN 激活张量**
2. Attention 中间张量（Q/K/V / softmax）
3. VAE 对整段视频一次性 encode / decode

---

## 2. FFN Chunk（最关键的突破）

### 2.1 什么是 FFN？

标准 Transformer FFN 结构：

```
[B, N, C]
 → Linear(C → ffn_dim)
 → GELU
 → Linear(ffn_dim → C)
```

其中会生成一个**巨大的中间张量**：

```
[B, N, ffn_dim]
```

这正是 DiT 在高分辨率 + 多帧场景下最容易 OOM 的原因。

---

### 2.2 FFN Chunk 的核心思想

**沿 token 维度 N 分块计算 FFN**：

```
[B, N, C]
 → 按 N 切成 K 份
 → 每一块独立走完整 FFN
 → 最后拼回
```

结果：

* **峰值显存 ≈ 原来的 1 / chunks**
* 数值结果完全一致（没有近似）
* 对画质 **零影响**

---

### 2.3 实际实现方式

运行时 patch 所有 FFN block：

* 替换 `ffn.forward`
* 沿 token 维 `N` 做 chunk
* 支持 auto 或固定 chunks

典型日志：

```
[FFN-Chunk] patched 30 FFN blocks with auto-chunking
[FFN-Chunk] auto select chunks=8 for N=11520
```

---

### 2.4 为什么 FFN Chunk 如此有效？

* FFN 对每个 token **完全独立**
* 没有 token-token 依赖
* 是 Transformer 中 **最安全、最适合 chunk 的部分**

> 实测中，**仅 FFN chunk 就能降低 30%–60% 的峰值显存**。

---

## 3. N 是什么？为什么它决定一切？

### 3.1 N 的定义（Token 数）

在 DreamID-V / Wan 系列中：

```
N = T × (H / stride_h) × (W / stride_w)
```

其中：

* `T`：一次处理的帧数（chunk 内）
* `H, W`：输入分辨率
* `stride`：patch / VAE stride（通常 8 或 16）

例子（576×1024，stride=8，T=16）：

```
H' = 576 / 8 = 72
W' = 1024 / 8 = 128
N ≈ 16 × 72 × 128 = 147,456（原始）
```

模型内部会再做 packing / 压缩，日志中常见：

```
N ≈ 11k – 28k
```

---

### 3.2 为什么 N 这么重要？

* **FFN 显存 ∝ N × ffn_dim**
* Attention 也随 N 增长（虽然有优化）
* N 一大，OOM 风险指数级上升

---

## 4. Chunk 数量的推荐计算公式

### 4.1 实用经验公式（工程向）

```
recommended_chunks = ceil(N / 4000)
```

并限制在合理区间：

```
chunks = min(max(recommended_chunks, 2), 16)
```

---

### 4.2 分辨率 / 帧数 / Chunk 推荐表

| 分辨率   | 单 chunk 帧数 | 典型 N    | 推荐 chunk |
| ----- | ---------- | ------- | -------- |
| 512p  | 16         | ~6k     | 2–4      |
| 720p  | 16         | ~9k     | 4–6      |
| 1024p | 16         | ~11–14k | 6–8      |
| 1280p | 16         | ~18–22k | 8–12     |
| 1280p | 41         | ~25–28k | 8–16     |

> **经验甜点值**：
> 👉 `chunks = 8` 覆盖 1024p–1280p 的绝大多数场景

---

## 5. VAE Encode 的隐藏 OOM 来源

### 5.1 原始问题

原始实现通常是：

```python
vae.encode(整段视频)
```

问题：

* 瞬时中间张量巨大
* CUDA allocator 冷启动 + 碎片
* 很容易在 `F.pad` / conv 中 OOM

---

### 5.2 修复：时间维微批（Temporal Micro-Batch）

现在的策略是：

```
[C, T, H, W]
 → 沿 T 切小段
 → 每段单独 encode
 → 自动 OOM fallback（16 → 8 → 4 → 2 → 1）
```

特点：

* 编码结果默认放在 **CPU**
* 每段结束立即释放 GPU 中间张量
* 冷启动 OOM 基本消失

---

### 5.3 修复：VAE Decode 避免逐帧 cat

原始 decode（危险）：

```python
out = cat(out, out_i)  # 每一步都会复制旧 tensor
```

优化后：

```python
outs = []
for each frame:
    outs.append(out_i)
out = torch.cat(outs, dim=2)
```

效果：

* 显存峰值显著下降
* 避免 allocator 碎片
* 输出完全一致

---

## 6. Warmup Pass：把“1 秒技巧”正式化

### 6.1 现象来源

老用户经验：

> “先跑 1 秒 / 1 帧，再跑正式视频就不 OOM 了”

本质原因：

* CUDA allocator 冷启动
* kernel / Triton / cuDNN 初始化
* Attention / FFN autotune

---

### 6.2 正式 Warmup 设计

Warmup Pass 特点：

* 与正式 pipeline 完全一致
* 同分辨率、同 backend
* 极短时长（如 1 秒）
* 极少 step（如 4）
* **不保存任何输出**
* **不调用 empty_cache**

结果：

* allocator 保持“热态”
* 第一个正式 chunk 不再慢 / 不稳定

---

## 7. Cache 清理原则（非常重要）

### 正确规则：

| 位置                    | 是否清 cache |
| --------------------- | --------- |
| FFN / Attention 内     | ❌         |
| 推理 step 中             | ❌         |
| VAE encode / decode 中 | ❌         |
| **chunk 之间**          | ✅（一次）     |

原因：

* 过度清 cache 会破坏预热状态
* 只在段间清一次，既安全又高效

---

## 8. 实测效果（真实数据）

**1280p，5 秒视频：**

| 项目          | 优化前    | 优化后     |
| ----------- | ------ | ------- |
| 冷启动         | OOM    | 正常      |
| 峰值显存        | ~17GB+ | ~15.4GB |
| 推理阶段显存      | ~99%   | ~8GB    |
| 576×1024 5s | 20+ 分钟 | ~8.5 分  |
| 720×1280 5s | OOM    | ~18.5 分 |
| 画质          | –      | 完全一致    |

---

## 9. 总结

这套优化之所以有效，是因为：

1. **FFN chunk** 切掉最大显存源头
2. **VAE 时间分段** 消除冷启动峰值
3. **Warmup pass** 稳定 allocator / kernel
4. **段间 cache 清理** 防泄漏但不破坏热态

最终效果是把 DreamID-V 从：

> “靠运气勉强跑”

变成：

> **“可预测、可调优、可扩展的工程系统”**


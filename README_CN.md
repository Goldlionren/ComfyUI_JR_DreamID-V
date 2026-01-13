# ComfyUI_JR_DreamID-V

<p align="center">
  <img src="https://img.shields.io/badge/ComfyUI-Plugin-blue" alt="ComfyUI Plugin">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License">
  <img src="https://img.shields.io/badge/Python-3.8+-blue" alt="Python">
</p>

**ComfyUI_JR_DreamID-V** 是基于原项目
👉 **HM-RunningHub / ComfyUI_RH_DreamID-V**
的 **JR 维护分支（Fork）**。

本项目同样基于 [DreamID-V](https://github.com/bytedance/DreamID-V)，用于在 ComfyUI 中实现高保真视频人脸交换，但 **重点面向非 4090 / 5090 用户，提升低显存与多设备环境的可用性与稳定性**。

---

## ✨ JR Fork 的目标与改进方向

> **让更多人用得起、用得稳，而不是只为顶级显卡服务**

相较原版，本 JR 分支的主要目标是：

* ✅ 支持 **16GB 显存 GPU**（如 RTX 4060 Ti / RTX 4080）
* ✅ 支持 **双 GPU 分工**（例如 T5 → 第二张卡）
* ✅ 支持 **CPU / GPU 混合 Offload**
* ✅ 修复多个在中低显存环境下常见的 OOM / 兼容性问题
* ✅ 在保持兼容原有 workflow 的前提下，提供更灵活的设备配置
* 🕺 **DWPose（ONNX + GPU 加速，JR 增强）**  
  使用 DWPose（ONNXRuntime）替代原有 MediaPipe 姿态提取流程，  
  在复杂动作视频中稳定性与成功率显著提升，支持 CUDA / TensorRT 加速。
---

## ✨ 功能特点

* 🎭 **高保真人脸交换**：基于 Diffusion Transformer 的视频人脸交换技术
* 🎬 **视频驱动**：支持使用视频作为动作驱动源
* 🖼️ **参考图像**：使用单张人脸图像作为身份参考
* 🔧 **ComfyUI 集成**：无缝集成至 ComfyUI 工作流
* 🧠 **低显存友好（JR Fork）**：T5 / 主模型 / VAE 可分设备加载
* 🕺 **DWPose（ONNX + GPU 加速，JR 增强）**  
  使用 DWPose（ONNXRuntime）替代原有 MediaPipe 姿态提取流程，  
  在复杂动作视频中稳定性与成功率显著提升，支持 CUDA / TensorRT 加速。
---

## 📋 节点说明

本插件提供两组节点（**完全兼容旧 workflow**）：

### ✅ JR 节点（推荐新用户使用）

| 节点名称                   | 功能说明                      |
| ---------------------- | ------------------------- |
| `JR_DreamID-V_Loader`  | 加载 DreamID-V 模型管线（支持设备选择） |
| `JR_DreamID-V_Sampler` | 执行视频人脸交换采样                |
| `JR_DreamID-V_LongVideo_Sampler` | **长视频分块采样（推荐用于长视频，避免 OOM）** |

### 🔁 Legacy 节点（兼容旧 workflow）

| 节点名称                           | 功能说明               |
| ------------------------------ | ------------------ |
| `RunningHub_DreamID-V_Loader`  | 原始 Loader（Legacy）  |
| `RunningHub_DreamID-V_Sampler` | 原始 Sampler（Legacy） |

> 💡 **建议新建工作流时使用 JR 节点，旧 workflow 无需修改即可继续使用。**

---

## ⚡ DreamID-V Wan-Faster 后端（JR 集成）

本 JR Fork 已集成 **DreamID-V Wan-1.3B-Faster** 推理后端，
在保持身份一致性的前提下，**显著降低采样步数并提升整体推理速度**。

### Wan-Faster 推荐参数

| 参数名 | 推荐值 | 说明 |
|------|-------|------|
| `backend` | `wan_faster` | 需在 Sampler 中手动选择 |
| `sampling_steps` | **12** | Faster 模型针对短步数训练 |
| `fps`（长视频） | **16** | 速度 / 质量平衡最佳 |
| `sample_solver` | `unipc` | **必须使用** |
| `frame_num` | 81 | 与标准 DreamID-V 一致 |
| `overlap_frames` | 8–12 | 长视频推荐 |

> ⚠️ 在 `wan_faster` 模式下使用 20+ steps **不会提升质量，只会变慢**。
---

## 🎞️ 长视频分块采样（JR 增强）

`JR_DreamID-V_LongVideo_Sampler` 用于处理**长视频**（帧数过多时单次推理容易 OOM）。  
它会把输入视频按固定帧数分块，逐块推理并把输出帧临时写入磁盘，最后合并成一个完整 MP4 输出,显存将不再是你的瓶颈,理论上可以理解为无限长度,最终边界是时间和硬盘。

### 关键参数说明

* **`frame_num`**：**每个 chunk 的帧数（chunk 大小就在这里设置）**。  
  例如：总 1620 帧，`frame_num=81` → 约 20 个 chunk。
> 💡 **Wan-Faster 建议**：  
> 使用 `backend=wan_faster` 时，推荐将 `fps` 设置为 **16**，以获得最佳速度与稳定性。


* **`fps`**（仅 LongVideo 节点）：
  * `-1`（默认）：跟随源视频 FPS（不做重采样）
  * `>= 1`：在生成 pose/mask 与推理之前执行**按时间的 FPS 重采样**
    * 保持**原始时长**（不会慢动作）
    * 显著减少模型需要处理的帧数
    * 对高 FPS 源视频提速明显（例如 60 → 24）

* **`overlap_frames`**：用于提升 chunk 之间的稳定性（warm-up 重叠帧）。  
  对于 `i>0` 的 chunk，会从上一段末尾借 `overlap_frames` 帧作为“稳定输入”，
  但在输出时会**丢弃重叠部分**，避免重复帧。

* **`return_frames_as_images` / `max_frames_to_return`**：可选返回帧张量（长视频不建议开启）。  
  更推荐使用 `frames_dir` 作为下游输入。

* **`keep_temp`**：是否保留中间 chunk 文件（便于调试，默认清理）。

### 输出说明

* **`video`**：最终合成的 MP4（音频从原始输入视频 mux）
* **`frames_dir`**：合并后的 PNG 帧目录（`frame_%08d.png`），推荐下游使用
* **`frames`**：可选 IMAGE 批量输出（受 `max_frames_to_return` 限制）

### 依赖要求

该节点依赖 **FFmpeg / FFprobe** 做探测、裁剪与编码，请确保系统 `PATH` 中可直接调用：
`ffmpeg`、`ffprobe`。

---
## ⚠️ 重要说明：Wan-Faster 必须安装 FlashAttention 2

> **如果你使用 `wan_faster` 后端，请务必阅读本节**

### 为什么必须安装 FlashAttention 2？

`wan_faster` 后端在注意力实现中 **强依赖 FlashAttention 2（FA2）**：

* ❌ **不是可选**
* ❌ **不会自动安装**
* ❌ 缺失会直接运行崩溃

常见报错包括：

```
AssertionError: FLASH_ATTN_2_AVAILABLE
```

或：

```
CUDA error: no kernel image is available for execution on the device
```

---

### FlashAttention 2 支持的显卡

FA2 仅支持 **Ampere 及更新架构**：

| 显卡              | 架构     | 是否支持  |
| --------------- | ------ | ----- |
| RTX 3060        | SM 8.6 | ✅ 支持  |
| RTX 3080 / 3090 | SM 8.6 | ✅     |
| RTX 40 系列       | SM 8.9 | ✅     |
| RTX 20 系列       | SM 7.5 | ❌ 不支持 |

检测方法：

```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
```

---

### FlashAttention 2 安装方式（强烈推荐）

推荐环境（已实测）：

* Python **3.10-12**
* CUDA **12.x / 13.0**
* PyTorch CUDA 版本

安装命令：

```bash
pip install flash-attn==2.8.2 --no-build-isolation
```

安装后验证：

```bash
python - <<EOF
import torch
from flash_attn import flash_attn_func
q = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)
o = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
print("FlashAttention 正常:", o.shape)
EOF
```

测试通过即可放心使用 `wan_faster`。

如果上述方法遇到困难,也可以直接使用: https://github.com/Goldlionren/AI-windows-whl.git
里面的whl,会容易很多.
---

### 后端选择建议

| 后端           | 是否需要 FA2 | 说明      |
| ------------ | -------- | ------- |
| `wan`        | ❌ 不需要    | 兼容性最好   |
| `wan_faster` | ✅ 必须     | 更快、更省步数 |

---


## 🕺 DWPose 姿态后端（JR 增强功能）

JR Fork 已将姿态提取后端升级为 **DWPose（基于 ONNXRuntime）**，
用于替代原有的 MediaPipe FaceMesh 流程。

### 为什么使用 DWPose？

* ✅ 对快速 / 复杂动作更稳定
* ✅ 显著减少 “no pose detected” 错误
* ✅ 支持 **ONNX Runtime GPU 加速（CUDA / TensorRT）**
* ✅ 与 PyTorch 设备配置完全解耦

---

### 后端行为说明

* **默认姿态后端**：`dwpose`
* **自动回退机制**：ONNXRuntime / GPU 不可用时回退至 MediaPipe
* **设备完全独立**：
  * T5 可运行在 **CPU**
  * DWPose 仍可运行在 **GPU**
  * 两者互不影响

---

### DWPose 所需模型（ONNX）

请将以下模型放置于：

```
ComfyUI/models/DreamID-V/pose/models/
├── dw-ll_ucoco_384.onnx
└── yolox_l.onnx
```

⚠️ 模型文件 **不包含在仓库中**。

---

### 自动下载（可选）

JR Fork 支持 **自动下载 DWPose ONNX 模型**。

可通过环境变量启用：

```bash
DREAMIDV_AUTO_DOWNLOAD_DWPOSE=1
```

若未启用，将给出清晰提示，指明缺失文件与放置路径。

---

### ONNXRuntime 加速说明

* 支持的执行后端：
  * `CUDAExecutionProvider`
  * `TensorrtExecutionProvider`（如环境支持）
  * `CPUExecutionProvider`（回退）

运行时日志示例：

```
[DWPose] det providers : ['CUDAExecutionProvider', 'CPUExecutionProvider']
[DWPose] pose providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

---

## 🚀 使用方法

1. 安装 ComfyUI（建议 Python ≥ 3.10）
2. 将本仓库克隆至：

```
ComfyUI/custom_nodes/ComfyUI_JR_DreamID-V
```

3. 安装依赖：

```bash
pip install -r requirements.txt
```

4. 下载 DreamID-V 与 DWPose 所需模型
5. 启动 ComfyUI

> 💡 **说明**：  
> 即使在 Loader 中将 **T5 设置为 `cpu`**，  
> DWPose 姿态提取仍可通过 ONNXRuntime 使用 GPU，两者互不影响。

---

## 🖥️ 系统要求

* 操作系统：Windows / Linux
* GPU：NVIDIA（推荐 ≥16GB 显存）
* Python：3.10+
* PyTorch：CUDA 版本
* **ONNX Runtime**：
  * `onnxruntime`（CPU）
  * `onnxruntime-gpu`（推荐，支持 GPU 加速）

---

## 🔀 JR Fork 核心改进点

相较于原版 DreamID-V：

* ✅ 使用 DWPose（ONNX）替代 MediaPipe 姿态提取
* ✅ 支持 GPU 加速姿态识别（CUDA / TensorRT）
* ✅ 明确区分 T5 / Pose / UNet 的运行设备
* ✅ 在真实视频场景下稳定性显著提升

---

## 🛠️ 安装指南

### 方法一：通过 ComfyUI Manager 安装（如后续支持）

1. 安装 [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)
2. 搜索 `ComfyUI_JR_DreamID-V`
3. 点击安装

### 方法二：手动安装（推荐）

1. 进入 ComfyUI 的 `custom_nodes` 目录：

```bash
cd ComfyUI/custom_nodes
```

2. 克隆 JR 仓库：

```bash
git clone https://github.com/<你的GitHub用户名>/ComfyUI_JR_DreamID-V.git
```

3. 安装依赖：

```bash
cd ComfyUI_JR_DreamID-V
pip install -r requirements.txt
```

---

## 📦 模型下载与配置

模型准备方式 **与原项目完全一致**。

### 1. Wan2.1-T2V-1.3B 基础模型

下载地址：
🤗 [https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)

放置路径：

```
ComfyUI/models/Wan/Wan2.1-T2V-1.3B/
├── models_t5_umt5-xxl-enc-bf16.pth
├── Wan2.1_VAE.pth
└── google/umt5-xxl/
```

### 2. DreamID-V 模型

下载地址：
🤗 [https://huggingface.co/XuGuo699/DreamID-V](https://huggingface.co/XuGuo699/DreamID-V)

放置路径：

```
ComfyUI/models/DreamID-V/
└── dreamidv.pth
```

### 3. DreamID-V Wan-Faster 模型（`wan_faster` 必需）

如果你计划使用 **Wan-Faster 后端**，还需要额外下载一个模型文件。

#### 下载地址

请从 DreamID-V 原项目下载 **`dreamidv_faster.pth`**：

👉 https://github.com/bytedance/DreamID-V  
（参考原仓库 README 中的 **Wan-1.3B-Faster** 说明）

> ⚠️ 本仓库 **不包含** `dreamidv_faster.pth`，  
> 请务必从原作者仓库自行下载。

#### 放置路径

放置路径：

```
ComfyUI/models/DreamID-V/
├── dreamidv.pth
└── dreamidv_faster.pth
```
#### 使用说明

- `dreamidv_faster.pth` **仅在使用以下配置时才会加载**：backend = wan_faster
- 标准模式（`wan`）仍然使用 `dreamidv.pth`
- Loader 会根据所选 backend 自动加载对应模型文件




---

## 🚀 使用方法（JR 推荐）

1. 添加 `JR_DreamID-V_Loader` 节点
2. 在 Loader 中选择 **T5 设备**：

   * `cuda:1`（推荐双卡用户）
   * `cuda:0`
   * `cpu`（显存极限场景）
3. 添加 `JR_DreamID-V_Sampler`
4. 连接输入：

   * **pipeline**
   * **video**
   * **ref_image**
5. 设置参数并运行

---

## 💻 系统要求（JR Fork 实测）

* **GPU**：

  * ✅ 推荐：RTX 4060 Ti / RTX 4080（16GB）
  * ✅ 支持双 GPU（T5 / 主模型分离）
* **Python**：3.8+
* **CUDA**：11.7+
* **ComfyUI**：最新版

> ⚠️ 显存越大体验越好，但 **JR Fork 不再强制要求 4090 / 5090**。

---
## 🚀 Wan-Faster 使用方式（强烈推荐）

1. 添加 `JR_DreamID-V_Loader`
2. 添加 `JR_DreamID-V_Sampler` 或 `JR_DreamID-V_LongVideo_Sampler`
3. 在 **Sampler 节点**中：
   * 设置 **`backend = wan_faster`**
   * 设置 **`sampling_steps = 12`**
   * 确保 **`sample_solver = unipc`**
4. （长视频）设置：
   * **`fps = 16`**
5. 运行工作流

### 重要说明

* `wan_faster` 内部 **不使用 pose 参考视频**
* 实际使用的参考输入为：
  * 原始视频
  * 人脸 mask 视频
  * 单张参考人脸图像
* 推理进度将通过 **ComfyUI 绿色进度条** 显示

---
---

## 📌 MediaPipe 说明（已变为可选）

在 JR Fork 中：

* 只要选择 **DWPose**
* 或使用 **wan_faster**
* 或 Python ≥ 3.12

➡ **MediaPipe 将完全不参与运行**

这意味着：

* 可以安全移除 MediaPipe依赖,以及相对应的模型
* 不影响最终效果
* 不影响 DWPose / Wan-Faster

---

## 📝 License & Fork 声明

* 本项目基于 **Apache License 2.0**
* 本仓库为以下项目的 **Fork 并独立维护版本**：

```
HM-RunningHub / ComfyUI_RH_DreamID-V
```

原始版权归原作者所有，本 JR 分支在 Apache 2.0 许可下进行修改与再发布。

---

## 🙏 致谢

* DreamID-V（Bytedance）
* Wan Team
* ComfyUI
* 原 RunningHub 项目作者
>特别感谢 DreamID-V 原作者引入的 **Wan-1.3B-Faster**
>模型与推理方案，使得在保持身份一致性的同时，
>能够以更少的采样步数实现显著加速。
---

## ⚠️ 免责声明

本项目仅供学习与研究使用，请遵守当地法律法规，勿用于非法或侵权用途。

---

<p align="center">
  如果这个 JR Fork 对你有帮助，欢迎 ⭐ Star 支持！
</p>

---

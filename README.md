# ComfyUI_JR_DreamID-V

<p align="center">
  <img src="https://img.shields.io/badge/ComfyUI-Plugin-blue" alt="ComfyUI Plugin">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License">
  <img src="https://img.shields.io/badge/Python-3.8+-blue" alt="Python">
</p>

**ComfyUI_JR_DreamID-V** is a **JR-maintained fork** of
👉 **HM-RunningHub / ComfyUI_RH_DreamID-V**.

This project is based on [DreamID-V](https://github.com/bytedance/DreamID-V) and provides high-fidelity video face swapping inside ComfyUI.
Unlike the original fork which primarily targets high-end GPUs (RTX 4090 / 5090), **this JR fork focuses on broader hardware accessibility and stability**.

---

## ✨ Goals of the JR Fork

> **Make DreamID-V usable for more people, not only top-tier GPUs.**

This fork is designed to:

* ✅ Support **16GB VRAM GPUs** (e.g. RTX 4060 Ti, RTX 4080)
* ✅ Support **dual-GPU setups** (e.g. T5 encoder on a second GPU)
* ✅ Support **CPU / GPU mixed offloading**
* ✅ Reduce OOM issues on mid-range hardware
* ✅ Preserve full compatibility with existing RunningHub workflows

---

## ✨ Features

* 🎭 **High-Fidelity Face Swapping**
  Video face swapping powered by Diffusion Transformer

* 🎬 **Video-Driven Motion**
  Use a video as the motion / pose driver

* 🖼️ **Reference Image Identity**
  Single face image as identity reference

* 🔧 **Native ComfyUI Integration**
  Seamlessly integrated into ComfyUI workflows

* 🧠 **Low-VRAM Friendly (JR Fork)**
  Flexible device placement for T5 / main model / VAE

* 🕺 **DWPose (ONNX, GPU-Accelerated) Pose Extraction (JR Fork)**  
  Replaces legacy MediaPipe pose extraction with **DWPose (ONNXRuntime)**.  
  Supports **CUDA / TensorRT acceleration**, significantly improving stability and accuracy on complex motion videos.

---

## 📋 Nodes

This plugin provides **two sets of nodes**, with **full backward compatibility**.

### ✅ JR Nodes (Recommended)

| Node Name              | Description                                 |
| ---------------------- | ------------------------------------------- |
| `JR_DreamID-V_Loader`  | Load DreamID-V pipeline (device-selectable) |
| `JR_DreamID-V_Sampler` | Run video face swapping                     |
| `JR_DreamID-V_LongVideo_Sampler` | Run long video face swapping via chunking (recommended for long videos)|

### 🔁 Legacy Nodes (Compatibility)

| Node Name                      | Description                       |
| ------------------------------ | --------------------------------- |
| `RunningHub_DreamID-V_Loader`  | Legacy loader (for old workflows) |
| `RunningHub_DreamID-V_Sampler` | Legacy sampler                    |


> 💡 **New workflows should use JR nodes. Existing workflows will continue to work without modification.**

---
## ⚡ DreamID-V Wan-Faster Backend (JR Integrated)

This fork integrates **DreamID-V Wan-1.3B-Faster** as an optional backend,
providing **significantly faster inference** with reduced sampling steps,
while maintaining identity fidelity.

### Recommended Settings (Wan-Faster)

| Parameter        | Recommended | Notes |
|------------------|-------------|-------|
| `backend`        | `wan_faster` | Must be explicitly selected in Sampler |
| `sampling_steps`| **12**        | Faster model is trained for short schedules |
| `fps` (LongVideo)| **16**        | Strong speed / quality balance |
| `sample_solver` | `unipc`       | **Required** (others not supported) |
| `frame_num`     | 81            | Same as standard DreamID-V |
| `overlap_frames`| 8–12          | Recommended for long videos |

> ⚠️ Using higher sampling steps (e.g. 20+) with `wan_faster` provides **no quality benefit** and only increases runtime.
---

## 🎞️ Long Video (Chunked) Sampler (JR Enhancement)

`JR_DreamID-V_LongVideo_Sampler` is designed for **long videos** that would otherwise OOM when processed as a single clip.
It splits the input into chunks, processes each chunk sequentially, writes intermediate frames to disk, and finally merges them into a single output video.

### Key Parameters

* **`frame_num`**: **Chunk size in frames**.  
  Example: total 1620 frames, `frame_num=81` → 20 chunks.
> 💡 **Wan-Faster Recommendation**:  
> When using `backend=wan_faster`, `fps=16` is strongly recommended for optimal speed/quality tradeoff.


* **`fps`** (LongVideo only):
  * `-1` (default): follow source video FPS (no resampling)
  * `>= 1`: **time-based FPS resampling before pose/mask and inference**
    * Keeps the **original duration** (no slow-motion)
    * Reduces the number of frames processed by the model
    * Significantly improves speed for high-FPS sources (e.g., 60 → 24)

* **`overlap_frames`**: Warm-up overlap for temporal stability.  
  For chunk `i>0`, prepend `overlap_frames` frames from the end of the previous chunk **for stability only**,
  but **drop overlapped frames from the output** to avoid duplicates.

* **`return_frames_as_images` / `max_frames_to_return`**: Optional frame tensor output.  
  Recommended to keep disabled for long videos; use `frames_dir` instead.

* **`keep_temp`**: Keep intermediate chunk files on disk for debugging.

### Outputs

* **`video`**: Final merged MP4 (audio is muxed from the original input video)
* **`frames_dir`**: Directory containing merged PNG frames (`frame_%08d.png`) for downstream processing
* **`frames`**: Optional IMAGE batch (guarded by `max_frames_to_return`)

### Requirements

This node uses **FFmpeg/FFprobe** for probing, cutting and encoding.  
Make sure `ffmpeg` and `ffprobe` are available in your system `PATH`.

---

## 🕺 DWPose Pose Backend (JR Enhancement)

This JR fork integrates **DWPose (ONNX-based)** as the **default pose extraction backend**,
replacing the legacy MediaPipe FaceMesh pipeline.

### Why DWPose?

* ✅ Much higher robustness on fast / complex motions
* ✅ Fewer "no pose detected" failures
* ✅ GPU-accelerated via **ONNX Runtime (CUDA / TensorRT)**
* ✅ Fully independent from PyTorch device placement

---

### Backend Behavior

* **Default backend**: `dwpose`
* **Automatic fallback**: MediaPipe (if ONNXRuntime or GPU is unavailable)
* **Device independence**:
  * T5 can run on **CPU**
  * DWPose can still run on **GPU**
  * No cross-interference between PyTorch and ONNXRuntime

---

### Required DWPose Models (ONNX)

Place the following ONNX models under:

```
ComfyUI/models/DreamID-V/pose/models/
├── dw-ll_ucoco_384.onnx
└── yolox_l.onnx
```

⚠️ These models are **NOT included** in the repository.

---

### Automatic Download (Optional)

JR fork supports **automatic download** of DWPose ONNX models.

Enable by setting the environment variable:

```bash
DREAMIDV_AUTO_DOWNLOAD_DWPOSE=1
```

If disabled, a clear error message will indicate which files are missing
and where to place them.

---

### ONNXRuntime Acceleration

* Supported providers:
  * `CUDAExecutionProvider`
  * `TensorrtExecutionProvider` (if available)
  * `CPUExecutionProvider` (fallback)

Runtime log example:

```
[DWPose] det providers : ['CUDAExecutionProvider', 'CPUExecutionProvider']
[DWPose] pose providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

---

## 🚀 Usage

1. Install ComfyUI (Python ≥ 3.10 recommended)
2. Clone this repository into:

```
ComfyUI/custom_nodes/ComfyUI_JR_DreamID-V
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download required DreamID-V and DWPose models
5. Launch ComfyUI

> 💡 **Note**:  
> Selecting `cpu` for **T5** does **NOT** affect DWPose.  
> Pose extraction runs via **ONNXRuntime** and can still use GPU acceleration.

---

## 🖥️ System Requirements

* OS: Windows / Linux
* GPU: NVIDIA (16GB VRAM recommended)
* Python: 3.10+
* PyTorch: CUDA-enabled build
* **ONNX Runtime**:
  * `onnxruntime` (CPU)
  * `onnxruntime-gpu` (recommended for GPU acceleration)

---

## 🔀 JR Fork Highlights

Compared to the original DreamID-V:

* ✅ DWPose (ONNX) replaces MediaPipe for pose extraction
* ✅ GPU-accelerated pose detection (CUDA / TensorRT)
* ✅ Clear separation of T5 / Pose / UNet devices
* ✅ Improved stability on real-world videos

---



## 🛠️ Installation

### Method 1: ComfyUI Manager (Future Support)

1. Install [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)
2. Search for `ComfyUI_JR_DreamID-V`
3. Install

### Method 2: Manual Installation (Recommended)

1. Navigate to ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
```

2. Clone the JR fork:

```bash
git clone https://github.com/<your-github-username>/ComfyUI_JR_DreamID-V.git
```

3. Install dependencies:

```bash
cd ComfyUI_JR_DreamID-V
pip install -r requirements.txt
```

---

## 📦 Model Downloads & Setup

Model preparation is **identical to the original project**.

### 1. Wan2.1-T2V-1.3B Base Model

Download:
🤗 [https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)

Directory layout:

```
ComfyUI/models/Wan/Wan2.1-T2V-1.3B/
├── models_t5_umt5-xxl-enc-bf16.pth
├── Wan2.1_VAE.pth
└── google/umt5-xxl/
```

### 2. DreamID-V Model

Download:
🤗 [https://huggingface.co/XuGuo699/DreamID-V](https://huggingface.co/XuGuo699/DreamID-V)

Directory:

```
ComfyUI/models/DreamID-V/
└── dreamidv.pth
```

### 3. DreamID-V Wan-Faster Model (Required for `wan_faster` backend)

If you plan to use the **Wan-Faster backend**, an additional model file is required.

#### Download

Download **`dreamidv_faster.pth`** from the official DreamID-V repository:

👉 https://github.com/bytedance/DreamID-V  
(See the **Wan-1.3B-Faster** section in the upstream README)

> ⚠️ This repository does **NOT** redistribute `dreamidv_faster.pth`.
> Please download it directly from the original authors.

#### Placement

Directory:

```
ComfyUI/models/DreamID-V/
├── dreamidv.pth
└── dreamidv_faster.pth
```

#### Usage Notes

- `dreamidv_faster.pth` is **only required** when using:
backend = wan_faster

- The standard backend (`wan`) continues to use `dreamidv.pth`
- The loader will automatically select the correct checkpoint
based on the selected backend

---

## 🚀 Usage (JR Recommended)

1. Add **`JR_DreamID-V_Loader`**
2. Select **T5 device**:

   * `cuda:1` (recommended for dual-GPU)
   * `cuda:0`
   * `cpu` (low-VRAM / fallback)
3. Add **`JR_DreamID-V_Sampler`**
4. Connect:

   * `pipeline`
   * `video`
   * `ref_image`
5. Configure parameters and run

---

## 💻 System Requirements (JR Fork)

* **GPU**:

  * ✅ RTX 4060 Ti / RTX 4080 (16GB tested)
  * ✅ Dual-GPU setups supported
* **Python**: 3.8+
* **CUDA**: 11.7+
* **ComfyUI**: Latest version

> ⚠️ Larger VRAM improves performance, but **4090 / 5090 are NOT required**.

---
## 🚀 Using Wan-Faster Backend (Recommended)

1. Add **`JR_DreamID-V_Loader`**
2. Add **`JR_DreamID-V_Sampler`** or **`JR_DreamID-V_LongVideo_Sampler`**
3. In the **Sampler** node:
   * Set **`backend` = `wan_faster`**
   * Set **`sampling_steps` = 12**
   * Ensure **`sample_solver` = unipc**
4. (LongVideo only) Set:
   * **`fps = 16`** (recommended)
5. Run the workflow

### Notes

* `wan_faster` **does not use pose reference video** internally.
* Reference inputs are limited to:
  * source video
  * face mask video
  * reference image
* Progress is reported via ComfyUI’s native green progress bar.

---

## 📝 License & Fork Statement

* Licensed under **Apache License 2.0**
* This repository is a **fork** of:

```
HM-RunningHub / ComfyUI_RH_DreamID-V
```

Original copyright belongs to the original authors.
All modifications in this repository are made under the terms of Apache-2.0.

---

## 🙏 Acknowledgements

* DreamID-V (ByteDance)
* Wan Team
* ComfyUI
* Original RunningHub project authors
>**Special thanks to the original DreamID-V authors for introducing the**
>**Wan-1.3B-Faster** **model and inference pipeline, which enables**
>**significantly faster generation with reduced sampling steps.**
---

## ⚠️ Disclaimer

This project is for **research and educational purposes only**.
Please comply with local laws and regulations. Do not use this project for illegal activities or rights-infringing purposes.

---

<p align="center">
  If you find this JR fork helpful, please consider giving it a ⭐ Star!
</p>

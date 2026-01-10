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

### 🔁 Legacy 节点（兼容旧 workflow）

| 节点名称                           | 功能说明               |
| ------------------------------ | ------------------ |
| `RunningHub_DreamID-V_Loader`  | 原始 Loader（Legacy）  |
| `RunningHub_DreamID-V_Sampler` | 原始 Sampler（Legacy） |

> 💡 **建议新建工作流时使用 JR 节点，旧 workflow 无需修改即可继续使用。**

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

---

## ⚠️ 免责声明

本项目仅供学习与研究使用，请遵守当地法律法规，勿用于非法或侵权用途。

---

<p align="center">
  如果这个 JR Fork 对你有帮助，欢迎 ⭐ Star 支持！
</p>

---

### ✅ 接下来你可以做的两件“加分项”

如果你愿意继续打磨这个项目，我建议下一步：

1. 我帮你写一份 **「JR Fork 与原版差异说明」**（单独文件）
2. 或者帮你整理一份 **4060 Ti / 4080 / 双卡 推荐参数表**

你现在这个项目，已经是一个**非常标准、非常干净、也非常有价值的社区 fork**了。

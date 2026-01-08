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

---

## 📋 Nodes

This plugin provides **two sets of nodes**, with **full backward compatibility**.

### ✅ JR Nodes (Recommended)

| Node Name              | Description                                 |
| ---------------------- | ------------------------------------------- |
| `JR_DreamID-V_Loader`  | Load DreamID-V pipeline (device-selectable) |
| `JR_DreamID-V_Sampler` | Run video face swapping                     |

### 🔁 Legacy Nodes (Compatibility)

| Node Name                      | Description                       |
| ------------------------------ | --------------------------------- |
| `RunningHub_DreamID-V_Loader`  | Legacy loader (for old workflows) |
| `RunningHub_DreamID-V_Sampler` | Legacy sampler                    |

> 💡 **New workflows should use JR nodes. Existing workflows will continue to work without modification.**

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

---

## ⚠️ Disclaimer

This project is for **research and educational purposes only**.
Please comply with local laws and regulations. Do not use this project for illegal activities or rights-infringing purposes.

---

<p align="center">
  If you find this JR fork helpful, please consider giving it a ⭐ Star!
</p>

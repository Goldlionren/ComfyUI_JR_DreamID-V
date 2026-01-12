## 🧩 DWPose 只使用 CPU / 不走 GPU（ONNXRuntime CUDA 问题）

> 适用于 **ComfyUI_JR_DreamID-V / DreamID-V (DWPose backend)**
> **⚠️ 在提交 Issue 前，请完整阅读并执行下面的自检步骤**

---

### 📌 问题描述

请简要描述你遇到的问题（例如：DWPose 明显很慢、日志显示 CPU、以前能用 GPU 现在不行等）：

```
（请填写）
```

---

### ✅ 正确行为（Expected）

在运行 DreamID-V / DWPose 时，日志中应看到：

```text
[DWPose] det providers : ['CUDAExecutionProvider', 'CPUExecutionProvider']
[DWPose] pose providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

---

### ❌ 实际行为（Actual）

如果你看到的是下面这种，则说明 **ONNXRuntime GPU 未生效**：

```text
[DWPose] det providers : ['CPUExecutionProvider']
[DWPose] pose providers: ['CPUExecutionProvider']
```

---

## 🚨 重要结论（请先读）

> **这是环境问题，不是代码 Bug。**

在绝大多数案例中：

* 同一份代码
* 在另一台机器 / 之前版本
* 是可以正常使用 GPU 的

问题来自 **ONNXRuntime CUDA / cuDNN / PATH / TensorRT 配置不完整**。

---

## 🧪 必做自检（未完成将直接关闭 Issue）

### 1️⃣ ONNXRuntime 是否支持 CUDA

请在 **ComfyUI 使用的同一个 Python 环境** 中执行：

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

⬇️ 粘贴完整输出：

```
（请粘贴）
```

✅ 正常应至少包含：

```text
CUDAExecutionProvider
```

---

### 2️⃣ CUDA Runtime DLL 是否存在（必须）

```powershell
where.exe cublasLt64_12.dll
```

⬇️ 输出结果：

```
（请粘贴）
```

❗ 如果找不到：

* 你 **没有安装 CUDA 12.x Runtime / Toolkit**
* PyTorch 自带 CUDA **不算**

---

### 3️⃣ cuDNN 9.x 是否存在（强烈建议）

```powershell
where.exe cudnn64_9.dll
```

⬇️ 输出结果：

```
（请粘贴）
```

---

### 4️⃣ 直接测试 ONNX Session Provider（关键）

请执行下面完整脚本（修改为你的 onnx 路径）：

```python
import onnxruntime as ort

onnx_det  = r"...\yolox_l.onnx"
onnx_pose = r"...\dw-ll_ucoco_384.onnx"

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

s1 = ort.InferenceSession(onnx_det, providers=providers)
s2 = ort.InferenceSession(onnx_pose, providers=providers)

print("DET providers :", s1.get_providers())
print("POSE providers:", s2.get_providers())
```

⬇️ 输出结果：

```
（请粘贴）
```

---

## ⚠️ 常见致命坑（请确认）

### ❌ 是否在 providers 里使用了 TensorRT？

```python
"TensorrtExecutionProvider"
```

如果 **系统没有安装 TensorRT（缺 nvinfer_10.dll）**，ONNXRuntime 会：

* TensorRT 初始化失败
* **整体 fallback 到 CPU**
* 即使 CUDA 可用也不会使用

✅ 正确做法（推荐）：

```python
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
```

---

## 🖥️ 环境信息（请填写）

* OS：Windows 10 / 11
* GPU 型号：（如 RTX 4090 / 4080S + 4060Ti）
* NVIDIA Driver 版本：
* CUDA 版本（系统安装）：
* cuDNN 版本：
* PyTorch 版本：
* onnxruntime 版本：

```
（请填写）
```

---

## ❓ 常见误解说明（重要）

* **PyTorch 能用 GPU ≠ ONNXRuntime 能用 GPU**
* ONNXRuntime CUDA **依赖系统 CUDA 12.x + cuDNN 9.x DLL**
* 这是 Windows 上最常见的坑之一

---

## ✅ Issue 处理原则

* 未完成上述自检步骤的 Issue **将被直接关闭**
* 请先确保这是 **环境问题已排查后的异常**
* 欢迎补充你最终的解决方式，帮助后续用户

---

### 📎 参考

* ONNXRuntime CUDA EP Requirements
  [https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)

---

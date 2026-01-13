import comfy.utils

import argparse
from datetime import datetime
import logging
import os
import sys
import warnings
import uuid
import subprocess

import math
import glob
import gc

import urllib.request
import shutil

warnings.filterwarnings('ignore')

import torch, random
import torch.distributed as dist
from PIL import Image, ImageOps

# classic backend
from .dreamidv_wan import DreamIDV as DreamIDV_WAN
from .dreamidv_wan.configs import WAN_CONFIGS as WAN_CONFIGS_WAN, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES

# faster backend
from . import dreamidv_wan_faster
from .dreamidv_wan_faster.configs import WAN_CONFIGS as WAN_CONFIGS_FASTER

from .dreamidv_wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from .dreamidv_wan.utils.utils import cache_video, cache_image, str2bool

import cv2
import numpy as np
from .pose.extract import process_dwpose

import folder_paths

import types

def _lazy_import_mediapipe_backend():
    """
    Lazy import MediaPipe backend to avoid breaking ComfyUI startup on Python 3.13.
    Only call this when MediaPipe backend is actually required.
    """
    try:
        from .express_adaption.media_pipe import FaceMeshDetector, FaceMeshAlign_dreamidv
        from .express_adaption.get_video_npy import get_video_npy
        return FaceMeshDetector, FaceMeshAlign_dreamidv, get_video_npy
    except Exception as e:
        raise RuntimeError(
            "[DreamID-V] MediaPipe backend is unavailable.\n"
            "- If you are on Python 3.13: MediaPipe is not supported, please use DWPose backend.\n"
            "- If you are on Python <= 3.12: install optional dependency: pip install \"mediapipe<0.10.30\".\n"
            f"Original error: {repr(e)}"
        ) from e


# ---------------- DWPose ONNX models (ComfyUI/models convention) ----------------
DWPOSE_FILES = {
    "dw-ll_ucoco_384.onnx": "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx",
    "yolox_l.onnx": "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx",
}

def _dwpose_models_dir():
    # ComfyUI/models/DreamID-V/pose/models
    return os.path.join(folder_paths.models_dir, "DreamID-V", "pose", "models")

def ensure_dwpose_models(auto_download: bool = False, timeout: int = 30) -> str:
    """Ensure DWPose ONNX models exist. Returns the models directory path."""
    models_dir = _dwpose_models_dir()
    os.makedirs(models_dir, exist_ok=True)

    missing = [fn for fn in DWPOSE_FILES.keys() if not os.path.exists(os.path.join(models_dir, fn))]
    if not missing:
        return models_dir

    if not auto_download:
        raise FileNotFoundError(
            "Missing DWPose model files:\n"
            + "\n".join([f"  - {fn}" for fn in missing])
            + "\nPlease place them under:\n"
            + f"  {models_dir}\n"
            + "Or enable auto-download (auto_download_dwpose=True)."
        )

    for fn in missing:
        url = DWPOSE_FILES[fn]
        dst = os.path.join(models_dir, fn)
        tmp = dst + ".download"
        try:
            print(f"[DreamID-V] DWPose model missing: {dst}")
            print(f"[DreamID-V] Downloading: {url}")
            with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)

            # basic sanity check (avoid HTML error page etc.)
            if os.path.getsize(tmp) < 1024 * 1024:
                raise RuntimeError("Downloaded file too small; likely invalid content.")

            os.replace(tmp, dst)
            print(f"[DreamID-V] Downloaded: {dst}")
        except Exception as e:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            raise RuntimeError(
                "Failed to auto-download DWPose model:\n"
                f"  - file : {fn}\n"
                f"  - url  : {url}\n"
                f"  - error: {repr(e)}\n"
                "You can download manually and place it under:\n"
                f"  {models_dir}"
            )

    return models_dir









try:
    from comfy_api.input_impl.video_types import VideoFromFile
except ImportError:
    VideoFromFile = None





def _which_bin(name: str) -> str:
    p = shutil.which(name)
    if not p:
        raise RuntimeError(f"Required binary not found in PATH: {name}")
    return p

def _run_cmd(cmd: list[str], timeout: int | None = None) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n\nSTDOUT:\n{p.stdout}\n\nSTDERR:\n{p.stderr}"
        )
    return (p.stdout or "").strip()

def _ffprobe_fps_and_total_frames(video_path: str) -> tuple[float, int]:
    """
    Returns (fps, total_frames). Uses ffprobe; falls back to counted frames if nb_frames is unavailable.
    """
    ffprobe = _which_bin("ffprobe")

    fps_txt = _run_cmd([
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-of", "default=nk=1:nw=1",
        video_path
    ])
    if "/" in fps_txt:
        a, b = fps_txt.split("/", 1)
        fps = float(a) / float(b)
    else:
        fps = float(fps_txt)

    nb_frames_txt = _run_cmd([
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "default=nk=1:nw=1",
        video_path
    ])

    total_frames = -1
    if nb_frames_txt and nb_frames_txt != "N/A":
        try:
            total_frames = int(nb_frames_txt)
        except Exception:
            total_frames = -1

    if total_frames <= 0:
        count_txt = _run_cmd([
            ffprobe, "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames",
            "-of", "default=nk=1:nw=1",
            video_path
        ])
        total_frames = int(count_txt)

    return fps, total_frames

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _soft_empty_cache():
    """
    Aggressive cleanup between chunks to prevent VRAM fragmentation / accumulation.
    """
    try:
        gc.collect()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        import comfy.model_management as mm
        mm.soft_empty_cache()
    except Exception:
        pass

def _cut_video_by_frame_range(src_video: str, dst_video: str, start_frame: int, frame_count: int) -> None:
    """
    Frame-accurate cut using select=between(n,start,end). Outputs a video with frames re-timestamped from 0.
    """
    ffmpeg = _which_bin("ffmpeg")
    _ensure_dir(os.path.dirname(dst_video))
    end_frame = start_frame + frame_count - 1
    vf = f"select='between(n\\,{start_frame}\\,{end_frame})',setpts=PTS-STARTPTS"
    _run_cmd([
        ffmpeg, "-v", "error", "-y",
        "-i", src_video,
        "-vf", vf,
        "-vsync", "0",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        dst_video
    ])

def _extract_png_frames_from_video(src_video: str, dst_dir: str) -> None:
    """
    Extract all frames from src_video into dst_dir/frame_%08d.png (0-based contiguous index after extraction).
    """
    ffmpeg = _which_bin("ffmpeg")
    _ensure_dir(dst_dir)
    # -start_number 0 ensures consistent indexing
    _run_cmd([
        ffmpeg, "-v", "error", "-y",
        "-i", src_video,
        "-vsync", "0",
        "-start_number", "0",
        os.path.join(dst_dir, "frame_%08d.png"),
    ])

def _encode_video_from_png_frames(frames_dir: str, fps: float, out_path: str) -> None:
    """
    Encode frames_dir/frame_%08d.png into out_path (no audio).
    """
    ffmpeg = _which_bin("ffmpeg")
    _ensure_dir(os.path.dirname(out_path))
    _run_cmd([
        ffmpeg, "-v", "error", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "frame_%08d.png"),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        out_path
    ])

def _mux_audio_from_source(video_no_audio: str, source_video: str, out_path: str) -> None:
    """
    Copy audio (if any) from source_video to video_no_audio.
    """
    ffmpeg = _which_bin("ffmpeg")
    ffprobe = _which_bin("ffprobe")
    _ensure_dir(os.path.dirname(out_path))

    has_audio = False
    try:
        probe_cmd = [
            ffprobe, "-v", "quiet", "-print_format", "json",
            "-show_streams", "-select_streams", "a:0", source_video
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            import json
            info = json.loads(result.stdout)
            if info.get("streams"):
                has_audio = True
    except Exception:
        has_audio = False

    if has_audio:
        cmd = [
            ffmpeg, "-v", "error", "-y",
            "-i", video_no_audio,
            "-i", source_video,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            out_path
        ]
    else:
        cmd = [
            ffmpeg, "-v", "error", "-y",
            "-i", video_no_audio,
            "-c:v", "copy",
            out_path
        ]
    _run_cmd(cmd)







def generate_pose_and_mask_videos(ref_video_path, ref_image_path, pose_backend='dwpose', auto_download_dwpose=False, dwpose_fallback=True):

    print("Starting online generation of pose and mask videos...")


    # Prefer DWPose (ONNX) if requested
    if str(pose_backend).lower() == 'dwpose':
        try:
            print('[DreamID-V] Using DWPose backend')
            temp_dir = os.path.join(folder_paths.get_temp_directory(), 'dreamidv')
            os.makedirs(temp_dir, exist_ok=True)
            video_name = os.path.splitext(os.path.basename(ref_video_path))[0]
            pose_output_path = os.path.join(temp_dir, video_name + '_pose.mp4')
            mask_output_path = os.path.join(temp_dir, video_name + '_mask.mp4')

            models_dir = ensure_dwpose_models(auto_download=bool(auto_download_dwpose))
            det_model_path = os.path.join(models_dir, 'yolox_l.onnx')
            pose_model_path = os.path.join(models_dir, 'dw-ll_ucoco_384.onnx')

            # Run DWPose extractor (requires onnxruntime)
            process_dwpose(
                input_video_path=ref_video_path,
                output_pose_path=pose_output_path,
                output_mask_path=mask_output_path,
                det_model_path=det_model_path,
                pose_model_path=pose_model_path,
            )
            return pose_output_path, mask_output_path
        except Exception as e:
            if not dwpose_fallback:
                raise
            print(f"[DreamID-V] DWPose failed, fallback to MediaPipe. Error: {repr(e)}")
            try:
                FaceMeshDetector, FaceMeshAlign_dreamidv, get_video_npy = _lazy_import_mediapipe_backend()
            except Exception as mp_e:
                raise RuntimeError(
                    "[DreamID-V] DWPose failed and MediaPipe fallback is unavailable.\n"
                    "Please disable dwpose_fallback or use Python <= 3.12 with mediapipe installed.\n"
                    f"DWPose error: {repr(e)}\n"
                    f"MediaPipe error: {repr(mp_e)}"
                ) from mp_e

    FaceMeshDetector, FaceMeshAlign_dreamidv, get_video_npy = _lazy_import_mediapipe_backend()
    detector = FaceMeshDetector()
    get_align_motion = FaceMeshAlign_dreamidv()

    CORE_LANDMARK_INDICES = [
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324,
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        1, 2, 5, 6, 48, 64, 94, 98, 168, 195, 197, 278, 294, 324, 327, 4, 24,
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
        263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
        468, 473, 55, 65, 52, 53, 46, 285, 295, 282, 283, 276, 70, 63, 105, 66, 107,
        300, 293, 334, 296, 336, 156,
    ]
    FACE_OVAL_INDICES = [
        10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
    ]
    CORE_LANDMARK_INDICES.extend(FACE_OVAL_INDICES)
    CORE_LANDMARK_INDICES = list(set(CORE_LANDMARK_INDICES))
    def save_visualization_video(landmarks_sequence, output_filename, frame_size, fps=30, mode='points'):
        width, height = frame_size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for {output_filename}")
            return
        print(f"Saving {mode} video to {output_filename}...")
        for frame_landmarks in landmarks_sequence:
            frame_image = np.zeros((height, width, 3), dtype=np.uint8)
            if mode == 'points':
                for landmark in frame_landmarks:
                    x, y = int(landmark[0]), int(landmark[1])
                    cv2.circle(frame_image, (x, y), radius=2, color=(255, 255, 255), thickness=-1)
            elif mode == 'mask':
                face_oval_points = frame_landmarks.astype(np.int32)
                cv2.fillConvexPoly(frame_image, face_oval_points, color=(255, 255, 255))
            video_writer.write(frame_image)
        video_writer.release()
        print("Video saving complete.")
    fps = cv2.VideoCapture(ref_video_path).get(cv2.CAP_PROP_FPS)
    face_results = get_video_npy(ref_video_path)
    video_name = os.path.basename(ref_video_path).split('.')[0]
    #kiki:
    # temp_dir = os.path.join(os.path.dirname(ref_video_path), 'temp_generated')
    temp_dir = os.path.join(folder_paths.get_temp_directory(), 'dreamidv')
    os.makedirs(temp_dir, exist_ok=True)
    print(f'try open ref_image_path:{ref_image_path}')
    image = Image.open(ref_image_path).convert('RGB')
    ref_image = np.array(image)
    _, ref_img_lmk = detector(ref_image)
    _, pose_addvis = get_align_motion(face_results, ref_img_lmk)
    width, height = face_results[0]['width'], face_results[0]['height']
 
    pose_output_path = os.path.join(temp_dir, video_name + '_pose.mp4')
    core_landmarks_sequence = pose_addvis[:, CORE_LANDMARK_INDICES, :]
    save_visualization_video(
        landmarks_sequence=core_landmarks_sequence,
        output_filename=pose_output_path,
        frame_size=(width, height),
        fps=fps,
        mode='points'
    )
    mask_output_path = os.path.join(temp_dir, video_name + '_mask.mp4')
    face_oval_sequence = pose_addvis[:, FACE_OVAL_INDICES, :]
    save_visualization_video(
        landmarks_sequence=face_oval_sequence,
        output_filename=mask_output_path,
        frame_size=(width, height),
        fps=fps,
        mode='mask'
    )
    return pose_output_path, mask_output_path


import inspect

def _pipeline_generate_compat(pipeline, text_prompt, ref_paths, **gen_kwargs):
    """
    Best-effort call pipeline.generate() across WAN and WAN_FASTER:
    only passes kwargs that the target generate() actually accepts.
    """
    fn = getattr(pipeline, "generate", None)
    if fn is None:
        raise RuntimeError("Pipeline has no generate()")

    try:
        sig = inspect.signature(fn)
        accepted = set(sig.parameters.keys())
        filtered = {k: v for k, v in gen_kwargs.items() if k in accepted}
    except Exception:
        filtered = dict(gen_kwargs)

    return fn(text_prompt, ref_paths, **filtered)



class RunningHub_DreamID_V_Loader:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "backend": (["wan", "wan_faster"], {"default": "wan"}),
                "main_device": (["cuda:0", "cuda:1"], {"default": "cuda:0"}),
                "t5_device": (["cpu", "cuda:0", "cuda:1"], {"default": "cpu"}),
            }
        }

    RETURN_TYPES = ('RH_DreamID-V_Pipeline', )
    RETURN_NAMES = ('DreamID-V Pipeline', )
    FUNCTION = "load"
    CATEGORY = "RunningHub/DreamID-V"
    OUTPUT_NODE = True

    def load(self, t5_device="cpu", backend="wan", main_device="cuda:0", **kwargs):
        task = 'swapface'
        ckpt_dir = os.path.join(folder_paths.models_dir, 'Wan', 'Wan2.1-T2V-1.3B')

        if backend == "wan_faster":
            dreamidv_ckpt = os.path.join(folder_paths.models_dir, 'DreamID-V', 'dreamidv_faster.pth')
            cfg0 = WAN_CONFIGS_FASTER[task]
        else:
            dreamidv_ckpt = os.path.join(folder_paths.models_dir, 'DreamID-V', 'dreamidv.pth')
            cfg0 = WAN_CONFIGS_WAN[task]

        # copy + inject t5_device
        if isinstance(cfg0, dict):
            cfg_dict = dict(cfg0)
            cfg_dict["t5_device"] = t5_device
            cfg = types.SimpleNamespace(**cfg_dict)
        else:
            cfg = cfg0
            setattr(cfg, "t5_device", t5_device)

        print(f"[Loader] backend={backend} main_device={main_device} t5_device={t5_device}")

        if backend == "wan_faster":
            # Faster 支持 device_id + t5_cpu (bool)
            device_id = 0 if str(main_device) == "cuda:0" else 1
            t5_cpu = (str(t5_device).lower() == "cpu")

            # Optional enhancement: pass t5_device_id so T5 can live on cuda:1
            # cpu  -> t5_cpu=True,  t5_device_id=None
            # cuda:X -> t5_cpu=False, t5_device_id=X
            t5_device_id = None
            try:
                if not t5_cpu and str(t5_device).startswith("cuda:"):
                    t5_device_id = int(str(t5_device).split(":")[1])
            except Exception:
                t5_device_id = None
 
            wan_swapface = dreamidv_wan_faster.DreamIDV(
                config=cfg,
                checkpoint_dir=ckpt_dir,
                dreamidv_ckpt=dreamidv_ckpt,
                device_id=device_id,
                rank=0,
                t5_fsdp=False,
                dit_fsdp=False,
                use_usp=False,
                t5_cpu=t5_cpu,
                t5_device_id=t5_device_id,
            )
            setattr(wan_swapface, "_jr_backend", "wan_faster")
            setattr(wan_swapface, "_jr_main_device", main_device)
            setattr(wan_swapface, "_jr_t5_device", t5_device)
        else:
            wan_swapface = DreamIDV_WAN(
                config=cfg,
                checkpoint_dir=ckpt_dir,
                dreamidv_ckpt=dreamidv_ckpt,
            )
            setattr(wan_swapface, "_jr_backend", "wan")

        return (wan_swapface,)


class RunningHub_DreamID_V_Sampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                #"type": (["Wan2.2 I2V", "Wan2.1 T2V"], ),
                "pipeline": ("RH_DreamID-V_Pipeline", ),
                "video": ("VIDEO", ),
                "ref_image": ("IMAGE", ),
                "size": (["832*480", "1280*720", "480*832", "720*1280", "custom"], {"default": "832*480"}),
                "frame_num": ("INT", {"default": 81, "min": 1, 'step': 1}),
                "sample_steps": ("INT", {"default": 20,}),
                # fps for output video (short sampler does not support auto)
                "fps": ("INT", {"default": 24, "min": 1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "pose_backend": (["dwpose", "mediapipe"], {"default": "dwpose"}),
                "auto_download_dwpose": ("BOOLEAN", {"default": False}),
                "dwpose_fallback": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "custom_width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8}),
                "custom_height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 8}),
            }
        }

    RETURN_TYPES = ('IMAGE', 'VIDEO')
    RETURN_NAMES = ('frames', 'video')
    FUNCTION = "sample"
    CATEGORY = "RunningHub/DreamID-V"

    OUTPUT_NODE = True

    def tensor_2_pil(self, img_tensor):
        i = 255. * img_tensor.squeeze().cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img

    def create_video_with_audio(self, frames_tensor, fps, source_video_path, output_path):
        """
        Create video from frames tensor and copy audio from source video.
        
        Args:
            frames_tensor: Tensor of shape (N, H, W, C) with values in [0, 1]
            fps: Frames per second
            source_video_path: Path to source video for audio extraction
            output_path: Output video file path
        """
        temp_video_path = output_path.replace('.mp4', '_temp.mp4')

        ffmpeg = _which_bin("ffmpeg")
        ffprobe = _which_bin("ffprobe")
        
        # Convert tensor to numpy frames
        frames_np = (frames_tensor.cpu().numpy() * 255).astype(np.uint8)
        num_frames, height, width, channels = frames_np.shape
        
        # Write frames to temp video using cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {temp_video_path}")
        
        for i in range(num_frames):
            # Convert RGB to BGR for cv2
            frame_bgr = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        video_writer.release()
        print(f"[DreamID-V] Wrote {num_frames} frames to temp video")
        
        # Check if source video has audio
        has_audio = False
        try:
            probe_cmd = [
                ffprobe, '-v', 'quiet', '-print_format', 'json',
                '-show_streams', '-select_streams', 'a:0', source_video_path
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                import json
                info = json.loads(result.stdout)
                if info.get('streams'):
                    has_audio = True
        except Exception as e:
            print(f"[DreamID-V] Could not probe audio: {e}")
        
        # Combine video with audio from source
        if has_audio:
            print(f"[DreamID-V] Copying audio from source video...")
            cmd = [
                ffmpeg, '-y',
                '-i', temp_video_path,
                '-i', source_video_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '18',
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0?',
                '-shortest',
                output_path
            ]
        else:
            print(f"[DreamID-V] No audio in source video, encoding video only...")
            cmd = [
                ffmpeg, '-y',
                '-i', temp_video_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '18',
                output_path
            ]
        
        try:
#            process = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
#            if process.returncode != 0:
#                print(f"[DreamID-V] FFmpeg error: {process.stderr}")
#                raise RuntimeError(f"FFmpeg failed: {process.stderr}")
            _run_cmd(cmd, timeout=300)
            print(f"[DreamID-V] Video created successfully: {output_path}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Video encoding timed out")
        finally:
            # Clean up temp file
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
        
        return output_path

    def create_video_object(self, video_path):
        """Create ComfyUI VIDEO object"""
        if VideoFromFile is not None:
            return VideoFromFile(video_path)
        else:
            # Fallback: return file path as string
            return video_path

    def sample(self, **kwargs):

        #kiki hardcode
        sample_shift = 5.0
        sample_solver = 'unipc'
        sample_guide_scale_img = 4.0

        pipeline = kwargs.get('pipeline')
        print(pipeline.config)
        pipeline.config.sample_fps = kwargs.get('fps')
        print(pipeline.config)
        sample_steps = kwargs.get('sample_steps')
        #self.pbar = comfy.utils.ProgressBar(sample_steps + 1)
        ref_video_path = kwargs.get('video').get_stream_source()
        ref_image = self.tensor_2_pil(kwargs.get('ref_image'))
        ref_image_path = os.path.join(folder_paths.get_temp_directory(), f'dreamidv_{uuid.uuid4()}.png')
        ref_image.save(ref_image_path)
        size = kwargs.get('size')
        if size == 'custom':
            custom_width = kwargs.get('custom_width', 832)
            custom_height = kwargs.get('custom_height', 480)
            size_tuple = (custom_width, custom_height)
        else:
            size_tuple = SIZE_CONFIGS[size]
        seed = kwargs.get('seed') ^ (2 ** 32)
        frame_num = kwargs.get('frame_num')

        # Ensure DWPose models exist (even if current pipeline still uses mediapipe)
        # This gives a clean error message and enables future DWPose backend switch.
        #auto_download = True
        auto_download = os.environ.get("DREAMIDV_AUTO_DOWNLOAD_DWPOSE", "0") == "1"

        try:
            ensure_dwpose_models(auto_download=auto_download)
        except Exception as e:
            # Do not hard-fail if you want to keep legacy mediapipe-only flow.
            # If you DO want to enforce presence, replace print(...) with "raise".
            print(f"[DreamID-V] DWPose model check warning: {e}")


        try:
            ref_pose_path, ref_mask_path = generate_pose_and_mask_videos(
                ref_video_path=ref_video_path,
                ref_image_path=ref_image_path,
                pose_backend=kwargs.get('pose_backend', 'dwpose'),
                auto_download_dwpose=kwargs.get('auto_download_dwpose', False),
                dwpose_fallback=kwargs.get('dwpose_fallback', True),
            )
        except:
            raise ValueError("Pose and mask video generation failed. no pose detected in the reference video.")
        text_prompt = 'change face'

        
        backend = getattr(pipeline, "_jr_backend", "wan")

        if backend == "wan_faster":
            # Faster 只吃 (video, mask, ref_image)
            ref_paths = [ref_video_path, ref_mask_path, ref_image_path]

            # Faster 只支持 unipc（否则 NotImplemented）
            sample_solver = "unipc"

            # 更贴近 Faster：如果用户没改 steps（你现在默认 20），建议降到 12
            if sample_steps == 20:
                sample_steps = 12
        else:
            ref_paths = [ref_video_path, ref_mask_path, ref_image_path, ref_pose_path]

        # ✅ create ProgressBar AFTER sample_steps is finalized
        self.pbar = comfy.utils.ProgressBar(sample_steps + 1)        
        self.update()

        generated = _pipeline_generate_compat(
            pipeline,
            text_prompt,
            ref_paths,
            size=size_tuple,
            frame_num=frame_num,
            shift=sample_shift,
            sample_solver=sample_solver,
            sampling_steps=sample_steps,
            guide_scale_img=sample_guide_scale_img,
            seed=seed,
            update_fn=self.update,      # classic 用得到；faster 会被过滤/或接受（按签名）
            offload_model=True,         # faster 用得到；classic 会被过滤
        )        

        print(f'generated video shape: {generated.shape}')
        
        # Convert to frames tensor (N, H, W, C) with values in [0, 1]
        frames = (generated.clamp(-1, 1).cpu().permute(1, 2, 3, 0) + 1.0) / 2.0
        
        # Create output video with audio from source
        fps = kwargs.get('fps')
        output_dir = folder_paths.get_output_directory()
        output_filename = f"dreamidv_{uuid.uuid4()}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        self.create_video_with_audio(frames, fps, ref_video_path, output_path)
        
        # Create VIDEO object
        video_obj = self.create_video_object(output_path)
        
        return (frames, video_obj)

    def update(self):
        self.pbar.update(1)

######按时间换算fps###########
def _resample_video_to_fps(src_video: str, fps_out: float, work_dir: str) -> str:
    """
    Resample video to target fps using time-based downsampling.
    Returns path to the resampled video.
    """
    ffmpeg = _which_bin("ffmpeg")
    _ensure_dir(work_dir)

    out_path = os.path.join(
        work_dir,
        f"resampled_fps{int(round(fps_out))}_{os.path.basename(src_video)}"
    )

    if os.path.exists(out_path):
        return out_path

    cmd = [
        ffmpeg, "-v", "error", "-y",
        "-i", src_video,
        "-vf", f"fps={fps_out}",
        "-fps_mode", "cfr",
        "-r", str(fps_out),
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        out_path
    ]
    _run_cmd(cmd)
    return out_path

#############################

class RunningHub_DreamID_V_LongVideo_Sampler(RunningHub_DreamID_V_Sampler):
    """
    Long video chunking sampler:
      - probe total frames
      - generate full pose/mask once (or fallback)
      - cut source/mask/pose into per-chunk short videos
      - run pipeline.generate per chunk
      - write per-chunk output frames to disk
      - merge into final mp4 and mux original audio

    Outputs:
      - frames (optional; risky for very long videos)
      - video
      - frames_dir (string path; recommended for downstream processing)
    """

    @classmethod
    def INPUT_TYPES(s):
        base = super().INPUT_TYPES()
        # LongVideo only: fps = -1 means "follow source video fps"
        # (We do NOT change short sampler's fps default)
        base["required"]["fps"] = ("INT", {"default": -1, "min": -1})        
        # extend required / optional
        base["required"].update({
            "return_frames_as_images": ("BOOLEAN", {"default": False}),
            "max_frames_to_return": ("INT", {"default": 300, "min": 0, "max": 100000}),
            "keep_temp": ("BOOLEAN", {"default": False}),
            # overlap (warm-up) frames: chunk i>0 will prepend these frames from previous chunk,
            # but we will DROP them from output to avoid duplicate frames.
            "overlap_frames": ("INT", {"default": 8, "min": 0, "max": 256}),
        })
        base["optional"].update({
            "temp_subdir": ("STRING", {"default": "dreamidv_long"}),
        })
        return base

    RETURN_TYPES = ('IMAGE', 'VIDEO', 'STRING')
    RETURN_NAMES = ('frames', 'video', 'frames_dir')
    FUNCTION = "sample_long"
    CATEGORY = "JR/DreamID-V"
    OUTPUT_NODE = True

    def _job_dirs(self, temp_subdir: str):
        root = os.path.join(folder_paths.get_temp_directory(), temp_subdir)
        job_id = uuid.uuid4().hex[:10]
        job_dir = _ensure_dir(os.path.join(root, f"job_{job_id}"))
        chunks_dir = _ensure_dir(os.path.join(job_dir, "chunks"))
        out_frames_dir = _ensure_dir(os.path.join(job_dir, "out_frames"))
        return job_dir, chunks_dir, out_frames_dir

    def _save_tensor_frames_to_dir(self, frames_tensor: torch.Tensor, out_dir: str, start_index: int) -> int:
        """
        frames_tensor: (N,H,W,C) in [0,1]
        Returns next start_index after saving.
        """
        _ensure_dir(out_dir)
        frames_np = (frames_tensor.detach().cpu().float().clamp(0, 1).numpy() * 255.0).astype(np.uint8)
        n = frames_np.shape[0]
        for i in range(n):
            img = Image.fromarray(frames_np[i])
            img.save(os.path.join(out_dir, f"frame_{start_index:08d}.png"))
            start_index += 1
        return start_index

    def sample_long(self, **kwargs):
        # reuse the same params & defaults as sample()
        pipeline = kwargs.get('pipeline')
        sample_steps = kwargs.get('sample_steps')
        fps_ui = kwargs.get('fps')
        size = kwargs.get('size')
        frame_num = kwargs.get('frame_num')  # chunk size
        seed_base = kwargs.get('seed') ^ (2 ** 32)
        overlap_frames = int(kwargs.get("overlap_frames", 8) or 0)

        return_frames_as_images = kwargs.get("return_frames_as_images", False)
        max_frames_to_return = int(kwargs.get("max_frames_to_return", 300))
        keep_temp = bool(kwargs.get("keep_temp", False))
        temp_subdir = kwargs.get("temp_subdir", "dreamidv_long")

        # progress bar: we cannot know total steps precisely (chunks * (steps+1)) until probing video
        # We'll create per-chunk progress bars instead.

        orig_video_path = kwargs.get('video').get_stream_source()
        ref_video_path = orig_video_path


        ref_image = self.tensor_2_pil(kwargs.get('ref_image'))
        ref_image_path = os.path.join(folder_paths.get_temp_directory(), f'dreamidv_{uuid.uuid4()}.png')
        ref_image.save(ref_image_path)

        if size == 'custom':
            custom_width = kwargs.get('custom_width', 832)
            custom_height = kwargs.get('custom_height', 480)
            size_tuple = (custom_width, custom_height)
        else:
            size_tuple = SIZE_CONFIGS[size]

        # Probe source video
        fps_src, total_frames = _ffprobe_fps_and_total_frames(ref_video_path)
        # fps_ui is INT; treat <0 (default -1) as "follow source fps"
        fps_ui = int(fps_ui) if fps_ui is not None else -1
        fps = float(fps_src) if fps_ui < 0 else float(fps_ui)

        # Time-based fps resampling BEFORE pose/mask and inference
        if fps_ui >= 1 and abs(fps - fps_src) > 1e-3:
            print(f"[DreamID-V][Long] Resampling video by time: {fps_src:.3f} -> {fps:.3f} fps")
            ref_video_path = _resample_video_to_fps(
                ref_video_path,
                fps_out=fps,
                work_dir=os.path.join(folder_paths.get_temp_directory(), "dreamidv_resample")
            )
            fps_src, total_frames = _ffprobe_fps_and_total_frames(ref_video_path)
            print(f"[DreamID-V][Long] Resampled video fps={fps_src:.3f} frames={total_frames}")

        # IMPORTANT: compute chunks AFTER resample (total_frames may change)
        num_chunks = int(math.ceil(total_frames / float(frame_num)))
        print(f"[DreamID-V][Long] source={orig_video_path}")
        if ref_video_path != orig_video_path:
            print(f"[DreamID-V][Long] infer_video={ref_video_path}")
        print(
            f"[DreamID-V][Long] fps(src)={fps_src:.3f} fps(out)={fps:.3f} "
            f"total_frames={total_frames} chunk_size={frame_num} chunks={num_chunks}"
        )

        # Ensure DWPose models exist (optional)
        auto_download = os.environ.get("DREAMIDV_AUTO_DOWNLOAD_DWPOSE", "0") == "1"
        try:
            ensure_dwpose_models(auto_download=auto_download)
        except Exception as e:
            print(f"[DreamID-V] DWPose model check warning: {e}")

        # Generate pose/mask ONCE for full video, then cut per chunk to match.
        try:
            ref_pose_path, ref_mask_path = generate_pose_and_mask_videos(
                ref_video_path=ref_video_path,
                ref_image_path=ref_image_path,
                pose_backend=kwargs.get('pose_backend', 'dwpose'),
                auto_download_dwpose=kwargs.get('auto_download_dwpose', False),
                dwpose_fallback=kwargs.get('dwpose_fallback', True),
            )
        except Exception:
            raise ValueError("Pose and mask video generation failed. no pose detected in the reference video.")

        # Setup temp job dirs
        job_dir, chunks_dir, out_frames_dir = self._job_dirs(temp_subdir)

        text_prompt = 'change face'
        sample_shift = 5.0
        sample_solver = 'unipc'
        sample_guide_scale_img = 4.0

        global_out_index = 0

        # Process chunks
        for ci in range(num_chunks):
            seed_eff = (seed_base + ci) & 0xFFFFFFFFFFFFFFFF
            start = ci * frame_num
            count = min(frame_num, total_frames - start)

            print(f"[DreamID-V][Long] chunk {ci+1}/{num_chunks} start_frame={start} count={count}")

            # Overlap scheme (Option 1):
            # - For ci>0, we prepend overlap_frames from previous chunk (warm-up),
            # - but drop those overlapped frames from the OUTPUT to avoid duplicates.
            # Compute the actual cut range and the number of frames to drop in output.
            if ci > 0 and overlap_frames > 0:
                cut_start = max(0, start - overlap_frames)
                # If start is very small (edge case), actual overlap may be smaller than requested
                drop = start - cut_start
                cut_count = count + drop
            else:
                cut_start = start
                drop = 0
                cut_count = count

            # Per-chunk temp videos
            chunk_dir = os.path.join(chunks_dir, f"chunk_{ci:05d}")
            _ensure_dir(chunk_dir)
            chunk_src = os.path.join(chunk_dir, "src.mp4")
            chunk_pose = os.path.join(chunk_dir, "pose.mp4")
            chunk_mask = os.path.join(chunk_dir, "mask.mp4")

            # Cut source/pose/mask by frame range (frame-accurate)
            _cut_video_by_frame_range(ref_video_path, chunk_src, cut_start, cut_count)
            _cut_video_by_frame_range(ref_pose_path, chunk_pose, cut_start, cut_count)
            _cut_video_by_frame_range(ref_mask_path, chunk_mask, cut_start, cut_count)

            # Update pipeline fps per UI
            pipeline.config.sample_fps = int(fps)

            # progress per chunk
            backend = getattr(pipeline, "_jr_backend", "wan")

            # decide per-chunk effective steps without mutating outer sample_steps
            steps_eff = sample_steps
            solver_eff = sample_solver

            if backend == "wan_faster":
                ref_paths = [chunk_src, chunk_mask, ref_image_path]
                solver_eff = "unipc"
                if steps_eff == 20:
                    steps_eff = 12
            else:
                ref_paths = [chunk_src, chunk_mask, ref_image_path, chunk_pose]

            # ✅ pbar after steps_eff is finalized
            self.pbar = comfy.utils.ProgressBar(steps_eff + 1)

            generated = _pipeline_generate_compat(
                pipeline,
                text_prompt,
                ref_paths,
                size=size_tuple,
                frame_num=cut_count,
                shift=sample_shift,
                sample_solver=solver_eff,
                sampling_steps=steps_eff,
                guide_scale_img=sample_guide_scale_img,
                seed=seed_eff,
                update_fn=self.update,
                offload_model=True,
            )



            # generated: (C, T, H, W)? existing code assumes (C,T,H,W) and converts to (T,H,W,C)
            frames = (generated.clamp(-1, 1).cpu().permute(1, 2, 3, 0) + 1.0) / 2.0

            # Drop overlapped warm-up frames from OUTPUT to avoid duplicates
            if drop > 0:
                frames_to_save = frames[drop:]
            else:
                frames_to_save = frames

            # Save frames to disk in a single global index space
            global_out_index = self._save_tensor_frames_to_dir(frames_to_save, out_frames_dir, global_out_index)

            # Cleanup to avoid OOM in next chunk
            try:
                del generated
                del frames
                del frames_to_save
            except Exception:
                pass
            _soft_empty_cache()

            # Remove per-chunk temp to save disk
            if not keep_temp:
                try:
                    shutil.rmtree(chunk_dir, ignore_errors=True)
                except Exception:
                    pass
            _soft_empty_cache()

        # Encode final video from PNG frames
        output_dir = folder_paths.get_output_directory()
        out_no_audio = os.path.join(output_dir, f"dreamidv_long_{uuid.uuid4()}.mp4")
        out_final = out_no_audio  # will be overwritten if audio mux is successful

        _encode_video_from_png_frames(out_frames_dir, fps=fps, out_path=out_no_audio)

        # mux audio from source
        try:
            out_muxed = os.path.join(output_dir, f"dreamidv_long_audio_{uuid.uuid4()}.mp4")
            _mux_audio_from_source(out_no_audio, orig_video_path, out_muxed)
            # if mux ok, prefer muxed output and remove no-audio
            out_final = out_muxed
            try:
                os.remove(out_no_audio)
            except Exception:
                pass
        except Exception as e:
            print(f"[DreamID-V][Long] audio mux failed, using no-audio video. Error: {repr(e)}")

        video_obj = self.create_video_object(out_final)

        # Optional: return frames as IMAGE batch (risk for long videos)
        frames_tensor_out = None
        if return_frames_as_images:
            if max_frames_to_return <= 0:
                print("[DreamID-V][Long] return_frames_as_images=True but max_frames_to_return<=0; skipping frames output.")
            elif total_frames > max_frames_to_return:
                print(f"[DreamID-V][Long] total_frames={total_frames} exceeds max_frames_to_return={max_frames_to_return}; skipping frames output.")
            else:
                # Load back PNGs to tensor (memory heavy; only allowed under limit)
                pngs = sorted(glob.glob(os.path.join(out_frames_dir, "frame_*.png")))
                imgs = []
                for p in pngs:
                    im = Image.open(p).convert("RGB")
                    arr = np.asarray(im).astype(np.float32) / 255.0
                    imgs.append(arr)
                if imgs:
                    frames_tensor_out = torch.from_numpy(np.stack(imgs, axis=0))

        # Cleanup temp job dir if requested.
        # IMPORTANT: keep out_frames_dir if we are returning it for downstream nodes.
        # When keep_temp=False, we only remove per-chunk intermediates (chunks_dir),
        # leaving out_frames_dir intact.
        if not keep_temp:
            try:
                shutil.rmtree(chunks_dir, ignore_errors=True)
            except Exception:
                pass

        return (frames_tensor_out, video_obj, out_frames_dir)





class JR_DreamID_V_Loader(RunningHub_DreamID_V_Loader):
    CATEGORY = "JR/DreamID-V"

class JR_DreamID_V_Sampler(RunningHub_DreamID_V_Sampler):
    CATEGORY = "JR/DreamID-V"

class JR_DreamID_V_LongVideo_Sampler(RunningHub_DreamID_V_LongVideo_Sampler):
    CATEGORY = "JR/DreamID-V"





NODE_CLASS_MAPPINGS = {
    # Legacy keys: keep for old workflows
    "RunningHub_DreamID-V_Loader": RunningHub_DreamID_V_Loader,
    "RunningHub_DreamID-V_Sampler": RunningHub_DreamID_V_Sampler,
    
    # JR keys: new workflows should use these
    "JR_DreamID-V_Loader": JR_DreamID_V_Loader,
    "JR_DreamID-V_Sampler": JR_DreamID_V_Sampler,
    "JR_DreamID-V_LongVideo_Sampler": JR_DreamID_V_LongVideo_Sampler,    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunningHub_DreamID-V_Loader": "RunningHub DreamID-V Loader (Legacy)",
    "RunningHub_DreamID-V_Sampler": "RunningHub DreamID-V Sampler (Legacy)",
    "JR_DreamID-V_Loader": "JR DreamID-V Loader",
    "JR_DreamID-V_Sampler": "JR DreamID-V Sampler",
    "JR_DreamID-V_LongVideo_Sampler": "JR DreamID-V Long Video Sampler (Chunked)",
}

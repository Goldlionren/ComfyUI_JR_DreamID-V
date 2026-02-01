# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm
import torchvision.transforms.functional as TF
from typing import List, Optional

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.na_resize import NaResize, DivisibleCrop, Rearrange
from torchvision.transforms import ToTensor,Normalize,Compose
from PIL import Image, ImageOps
import warnings


_THIS_DIR = os.path.dirname(__file__)

def ffn_chunked_forward(ffn_module, x, chunks: int, dim_threshold: int):
    # x expected: [B, N, C]
    if chunks <= 1 or x.dim() != 3:
        return ffn_module._orig_forward(x)

    B, N, C = x.shape
    if dim_threshold > 0 and N <= dim_threshold:
        return ffn_module._orig_forward(x)

    chunk_size = N // chunks
    y = torch.empty_like(x)
    for i in range(chunks):
        s = i * chunk_size
        e = (i + 1) * chunk_size if i < chunks - 1 else N
        y[:, s:e, :] = ffn_module._orig_forward(x[:, s:e, :])
    return y

def patch_ffn_chunking(model, chunks: int = 2, dim_threshold: int = 4096):
    for i, block in enumerate(model.blocks):
        if hasattr(block, "ffn"):
            ffn = block.ffn
        elif hasattr(block, "ff"):
            ffn = block.ff
        elif hasattr(block, "mlp"):
            ffn = block.mlp
        else:
            continue

        # 只 patch 一次
        if not hasattr(ffn, "_orig_forward"):
            ffn._orig_forward = ffn.forward

        def wrapped_forward(self_ffn, x):
            return ffn_chunked_forward(self_ffn, x, chunks=chunks, dim_threshold=dim_threshold)

        ffn.forward = types.MethodType(wrapped_forward, ffn)

class DreamIDV:

    def __init__(
        self,
        config,
        checkpoint_dir,
        dreamidv_ckpt,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        t5_device_id=None
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            dreamidv_ckpt (`str`):
                Path of DreamID-V dit checkpoint
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.t5_device_id = t5_device_id

        # ---- Decide T5 device (backward compatible) -------------------------------
        # Priority: explicit t5_device_id > t5_cpu > default to main device
        # NOTE: current Faster swapface path does not use T5 yet, but we keep this
        # for forward compatibility and for users who enable prompt-conditioned variants later.
        if t5_device_id is not None:
            try:
                if isinstance(t5_device_id, str):
                    s = t5_device_id.strip().lower()
                    if s.startswith("cuda:"):
                        self.t5_device = torch.device(s)
                    else:
                        self.t5_device = torch.device(f"cuda:{int(s)}")
                else:
                    self.t5_device = torch.device(f"cuda:{int(t5_device_id)}")
            except Exception:
                # fall back to main device if parsing fails
                self.t5_device = torch.device(f"cuda:{int(device_id)}")
        else:
            if t5_cpu:
                self.t5_device = torch.device("cpu")
            else:
                self.t5_device = torch.device(f"cuda:{int(device_id)}")

        try:
            logging.info(
                f"[DreamIDV-Faster] device_id={device_id} | t5_device={self.t5_device} | t5_cpu={t5_cpu} | t5_device_id={t5_device_id}"
            )
        except Exception:
            pass


        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {dreamidv_ckpt}")
        self.model = WanModel(
                             model_type=config.model_type,
                             dim=config.dim, 
                             ffn_dim=config.ffn_dim,
                             freq_dim=config.freq_dim,
                             in_dim=config.in_dim,
                             num_heads=config.num_heads,
                             num_layers=config.num_layers,
                             window_size=config.window_size,
                             qk_norm=config.qk_norm,
                             cross_attn_norm=config.cross_attn_norm,
                             eps=config.eps)
    
        logging.info(f"loading ckpt.")
        state = torch.load(dreamidv_ckpt, map_location="cpu")  # 1) 权重先落 CPU

        # 确保模型参数在哪里：你要推理用GPU就放GPU
        self.model.to(self.device)                               # <-- 放在 load_state_dict 前

        logging.info(f"loading state dict.")
        missing_keys, unexpected_keys = self.model.load_state_dict(state, strict=False)

        self.model.eval().requires_grad_(False)

        # debug info
        b0 = self.model.blocks[0]
        print("block0 type:", type(b0))
        print("has ffn:", hasattr(b0, "ffn"), "has ff:", hasattr(b0, "ff"), "has mlp:", hasattr(b0, "mlp"))
        print("ffn obj:", getattr(b0, "ffn", None))

        # Patch FFN chunking to reduce memory usage
        from .modules.ffn_chunk import patch_model_ffn_chunking
        patch_model_ffn_chunking(self.model, verbose=True)


        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

    def _encode_video_latent_chunked(
        self,
        video_cthw: torch.Tensor,
        device: torch.device,
        *,
        init_t_chunk: int = 16,
        min_t_chunk: int = 1,
        return_cpu: bool = True,
        tag: str = "ref",
    ) -> torch.Tensor:
        """
        Encode a single video tensor [C,T,H,W] with micro-batching on T to reduce peak VRAM.
 
        - Splits along T into chunks of size t_chunk.
        - On CUDA OOM, automatically reduces t_chunk and retries.
        - Optionally returns latents on CPU to minimize VRAM residency.
 
        Returns: latent tensor with time-dimension concatenated.
        """
        assert video_cthw.dim() == 4, f"expected [C,T,H,W], got {tuple(video_cthw.shape)}"
        C, T, H, W = video_cthw.shape

        # Early exit: empty
        if T <= 0:
            raise ValueError("video has zero frames (T==0)")
 
        # Choose starting chunk size
        t_chunk = max(min(init_t_chunk, T), min_t_chunk)
 
        def _empty_cuda_cache():
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                try:
                    torch.cuda.empty_cache()
                except Exception:
                   pass
 
        while True:
            lat_parts: List[torch.Tensor] = []
            try:
                with torch.no_grad(), amp.autocast(dtype=self.param_dtype):
                    for s in range(0, T, t_chunk):
                        e = min(T, s + t_chunk)
                        # slice [C, t, H, W]
                        chunk = video_cthw[:, s:e, :, :]
                        chunk_gpu = chunk.to(device=device, dtype=self.param_dtype, non_blocking=True)
 
                        # VAE expects list of [C,T,H,W]
                        z = self.vae.encode([chunk_gpu], device, return_cpu=return_cpu)[0]
                        # z is [Cz, t', h', w'] (Cz is z_dim)
                        lat_parts.append(z)
 
                        # release per-chunk GPU tensor ASAP
                        del chunk_gpu
                        _empty_cuda_cache()
 
                # concat along time dim=1 for [Cz, T', h', w']
                out = torch.cat(lat_parts, dim=1) if len(lat_parts) > 1 else lat_parts[0]

                # ------------------------------------------------------------
                # IMPORTANT: time-strided video VAE (stride_t usually=4) is NOT
                # linear in T when encoding chunks independently.
                # Example (stride_t=4):
                #   encode(13 frames) => latent T' = 4
                #   encode(6)+encode(6)+encode(1) => 2+2+1 = 5  (too long)
                #
                # If video OOM triggers chunking but mask does not, this causes
                # video latent T' != mask latent T', and later concat fails.
                #
                # Fix: clamp concatenated latent time length to the expected
                # length of encoding the (trimmed) full sequence.
                # ------------------------------------------------------------
                try:
                    stride_t = int(getattr(self, "vae_stride", (4, 8, 8))[0])
                    # input frames are trimmed in load_image_latent_ref_ip_video to:
                    # frames_num = floor((T-1)/4)*4 + 1  (to satisfy model constraint)
                    T_trim = ((T - 1) // 4) * 4 + 1
                    # expected latent length after temporal stride
                    expected_T_lat = ((T_trim - 1) // stride_t) + 1
                    got_T_lat = int(out.shape[1])
                    if got_T_lat != expected_T_lat:
                        if got_T_lat < expected_T_lat:
                            logging.warning(
                                f"[VAE-Encode-{tag}] latent T' shorter than expected "
                                f"(got {got_T_lat}, expected {expected_T_lat})"
                            )
                        else:
                            logging.warning(
                                f"[VAE-Encode-{tag}] latent T' longer than expected "
                                f"(got {got_T_lat}, expected {expected_T_lat}) -> truncating"
                            )
                            out = out[:, :expected_T_lat, ...].contiguous()
                except Exception:
                    pass

                return out
 
            except torch.OutOfMemoryError as oom:
                # cleanup
                for p in lat_parts:
                    try:
                        del p
                    except Exception:
                        pass
                lat_parts.clear()
                _empty_cuda_cache()
 
                if t_chunk <= min_t_chunk:
                    logging.error(f"[VAE-Encode-{tag}] OOM even at t_chunk={t_chunk}. Raising.")
                    raise
 
                # reduce and retry
                new_t = max(min_t_chunk, t_chunk // 2)
                logging.warning(f"[VAE-Encode-{tag}] OOM at t_chunk={t_chunk}, retry with t_chunk={new_t}")
                t_chunk = new_t
                continue
 
            finally:
                # best-effort gc
                try:
                    gc.collect()
                except Exception:
                     pass
                _empty_cuda_cache()


    def load_image_latent_ref_ip_video(self,paths: str, size, device, frame_num):
        # Load size.
        patch_size = self.patch_size
        vae_stride = self.vae_stride

        def is_image_or_video_by_extension(file_path):
            image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
            video_exts = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm"}
            
            ext = os.path.splitext(file_path)[1].lower()
            if ext in image_exts:
                return "image"
            elif ext in video_exts:
                return "video"
            else:
                return "unknown"

        # import pdb; pdb.set_trace()
        # Load image and video.
        ref_vae_latents = {
            "image": [],
            "video": [],
            "mask": [],
        }
        video_h = 0
        video_w = 0
        from decord import VideoReader
        for i, path in enumerate(paths):
            
            if is_image_or_video_by_extension(path) == "video" and i == 0:
                print('ori_path', path)
                vr = VideoReader(path)
                frames = []

                for idx in range(vr.__len__()):
                    frame = vr[idx].asnumpy()
                    frame = Image.fromarray(frame)
                    frames.append(frame)
                video_w = frames[0].size[0]
                video_h = frames[0].size[1]
            
                frames = frames[:frame_num]

                frames_num = (len(frames)-1)//4*4 +1
                frames = frames[:frames_num]

                video_transform=Compose(
                    [
                        NaResize(
                            resolution=math.sqrt(size[0] * size[1]), 
                            downsample_only=True,
                        ),
                        DivisibleCrop((vae_stride[1] * patch_size[1], vae_stride[2] * patch_size[2])),
                        Normalize(0.5, 0.5),
                        Rearrange("t c h w -> c t h w"),
                    ]
                )
                
                video_frames = video_transform(frames)
 
                # Encode video with micro-batching to reduce VRAM peak (avoid cold-start OOM)
                video_vae_latent = self._encode_video_latent_chunked(
                    video_frames,
                    device=device,
                    init_t_chunk=16,
                    min_t_chunk=1,
                    return_cpu=True,
                    tag="video",
                )
            
                ref_vae_latents["video"].append(video_vae_latent)
                # release CPU-side frame tensor as well
                del video_frames
                

            elif is_image_or_video_by_extension(path) == "video" and i == 1:
                print('mask_path', path)
                vr = VideoReader(path)
                frames = []
                for idx in range(vr.__len__()):
                    frame = vr[idx].asnumpy()
                    frame = Image.fromarray(frame)
                    frames.append(frame)
                video_w = frames[0].size[0]
                video_h = frames[0].size[1]
                
                frames = frames[:frame_num]

                frames_num = (len(frames)-1)//4*4 +1
                frames = frames[:frames_num]
                video_mask_transform=Compose(
                    [
                        NaResize(
                            resolution=math.sqrt(size[0] * size[1]), # 256*448, 480*832
                            downsample_only=True,
                        ),
                        DivisibleCrop((vae_stride[1] * patch_size[1], vae_stride[2] * patch_size[2])),
                        # Normalize(0.5, 0.5),
                        Rearrange("t c h w -> c t h w"),
                    ]
                )
                
                video_frames = video_mask_transform(frames) 
                video_vae_latent = self._encode_video_latent_chunked(
                    video_frames,
                    device=device,
                    init_t_chunk=16,
                    min_t_chunk=1,
                    return_cpu=True,
                    tag="mask",
                )
                ref_vae_latents["mask"].append(video_vae_latent)
                del video_frames

           

            elif is_image_or_video_by_extension(path) == "image" and i == 2:

                with Image.open(path) as img:
                    img = img.convert("RGB")
                    img_ratio = img.width / img.height
                    target_ratio = video_w / video_h
                    
                    if img_ratio > target_ratio: 
                        new_width = video_w
                        new_height = int(new_width / img_ratio)
                    else:  
                        new_height = video_h
                        new_width = int(new_height * img_ratio)

                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    delta_w = video_w - img.size[0]
                    delta_h = video_h - img.size[1]
                    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
                    new_img = ImageOps.expand(img, padding, fill=(255, 255, 255))

                # Transform to tensor and normalize.
                image_transform=Compose(
                    [
                        NaResize(
                            resolution=math.sqrt(size[0] * size[1]), 
                            downsample_only=True,
                        ),
                        DivisibleCrop((vae_stride[1] * patch_size[1], vae_stride[2] * patch_size[2])),
                        Normalize(0.5, 0.5),
                        Rearrange("t c h w -> c t h w"),
                    ]
                )
                new_img = image_transform([new_img])
                # Image path: still use the same chunked encoder for consistency (T==1)
                img_vae_latent = self._encode_video_latent_chunked(
                    new_img,
                    device=device,
                    init_t_chunk=1,
                    min_t_chunk=1,
                    return_cpu=True,
                    tag="image",
                )
                ref_vae_latents["image"].append(img_vae_latent)
            else:
                print("Unknown file type.")

        # ---- Safety: align video/mask latent time length ---------------------------------
        # Chunked VAE encode may produce different latent T' lengths between video and mask
        # (especially when one path OOM-retries with smaller t_chunk and the other doesn't).
        # This would crash later at torch.concat([y_i_v, msk]).
        if len(ref_vae_latents["video"]) > 0 and len(ref_vae_latents["mask"]) > 0:
            try:
                v0 = ref_vae_latents["video"][0]
                m0 = ref_vae_latents["mask"][0]
                tv = int(v0.shape[1])
                tm = int(m0.shape[1])
                if tv != tm:
                    tmin = min(tv, tm)
                    logging.warning(
                        f"[VAE-Encode-align] video/mask latent T' mismatch (video={tv}, mask={tm}) -> clamp to {tmin}"
                    )
                    ref_vae_latents["video"][0] = v0[:, :tmin, ...].contiguous()
                    ref_vae_latents["mask"][0]  = m0[:, :tmin, ...].contiguous()
            except Exception:
               pass
 

        ref_vae_latents["image"] = torch.cat(ref_vae_latents["image"], dim=0)
        ref_vae_latents["video"] = torch.cat(ref_vae_latents["video"], dim=0)
        ref_vae_latents["mask"] = torch.cat(ref_vae_latents["mask"], dim=0)

        # Keep latents on CPU here; caller can move to GPU as needed.
        # This avoids keeping large ref latents resident on VRAM during preprocessing.        
        
        return ref_vae_latents


        
    def generate(self,
                 input_prompt,
                 paths,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale_img=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 update_fn=None):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
             paths ([`str`])`:
                Reference paths for swap face.
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        device = self.device   
        dtype = self.param_dtype
        
        latents_ref = self.load_image_latent_ref_ip_video(paths, 
                                                          size, 
                                                          device,
                                                          frame_num,
                                                        )

        # Move ref latents to GPU only when needed (they are produced on CPU to reduce VRAM peak)
        latents_ref_video = latents_ref["video"].to(device=device, dtype=dtype, non_blocking=True)
        latents_ref_image = latents_ref["image"].to(device=device, dtype=dtype, non_blocking=True)
        msk = latents_ref["mask"].to(device=device, dtype=dtype, non_blocking=True)

        F = frame_num
        target_shape = (self.vae.model.z_dim, latents_ref_video.shape[1],
                        latents_ref_video.shape[2],
                        latents_ref_video.shape[3])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)



        # Use file-relative path to avoid CWD-dependent failures under ComfyUI
        context_path = os.path.join(_THIS_DIR, "context.pth")
        context = torch.load(context_path, map_location="cpu")
        context = [t.to(self.device) for t in context]
        
        y_i_v = latents_ref_video 
        img_ref = latents_ref_image

        arg_tiv = {
            'context': context, 
            'seq_len': seq_len,
            'y': [torch.concat([y_i_v, msk])],
  
            'img_ref': [img_ref]
            }

        
        arg_tv = {
            'context': context,
            'seq_len': seq_len, 
            'y': [torch.concat([y_i_v, msk])],
            'img_ref': [torch.zeros_like(img_ref)]
            }


        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=self.param_dtype,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            for i, t in enumerate(tqdm(timesteps)):
                if update_fn is not None:
                    # ComfyUI ProgressBar callback: accept either update_fn() or update_fn(step)
                    try:
                        update_fn(i)
                    except TypeError:
                        update_fn()

                timestep = [t]
                timestep = torch.stack(timestep)

                self.model.to(self.device)
                pos_tiv = self.model(latents, t=timestep, **arg_tiv)[0]
                pos_tv = self.model(latents, t=timestep, **arg_tv)[0]
                
                noise_pred = pos_tiv
                noise_pred += guide_scale_img * (pos_tiv - pos_tv) 
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0).to(self.param_dtype)]

            # finalize progress (optional): ensure bar reaches the end
            if update_fn is not None:
                try:
                    update_fn(len(timesteps))
                except TypeError:
                    update_fn()

            x0 = latents

            print("[DEBUG] autocast target dtype =", self.param_dtype)
            print("[DEBUG] input latent dtype =", x0[0].dtype)


            if offload_model:
                self.model.cpu()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Literal
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize
import math
from typing import List, Union
import torch
from PIL import Image
from torchvision.transforms import functional as TVF
from torchvision.transforms.functional import to_tensor
from einops import rearrange

def _normalize_interpolation(interp):
    """
    torchvision>=0.15 expects:
      - torchvision.transforms.InterpolationMode
      - OR a corresponding Pillow integer constant
    But some callers may pass string or OpenCV constants.
    """
    if interp is None:
        return InterpolationMode.BICUBIC

    # already correct enum
    if isinstance(interp, InterpolationMode):
        return interp

    # integers: could be PIL constant OR OpenCV interpolation
    if isinstance(interp, int):
        # OpenCV common values:
        # 0=INTER_NEAREST, 1=INTER_LINEAR, 2=INTER_CUBIC, 3=INTER_AREA, 4=INTER_LANCZOS4
        if interp == 0:
            return InterpolationMode.NEAREST
        if interp == 1:
            return InterpolationMode.BILINEAR
        if interp == 2:
            return InterpolationMode.BICUBIC
        if interp == 4:
            return InterpolationMode.LANCZOS
        # If it's a PIL resample int, torchvision accepts it as-is
        return interp

    # strings like "bicubic"/"bilinear"/"lanczos"
    if isinstance(interp, str):
        s = interp.strip().lower()
        mapping = {
            "nearest": InterpolationMode.NEAREST,
            "nearest-exact": InterpolationMode.NEAREST_EXACT,
            "bilinear": InterpolationMode.BILINEAR,
            "bicubic": InterpolationMode.BICUBIC,
            "box": InterpolationMode.BOX,
            "hamming": InterpolationMode.HAMMING,
            "lanczos": InterpolationMode.LANCZOS,
            # some callers use "area" (OpenCV style); torchvision has no AREA resize mode
            # map to a safe default
            "area": InterpolationMode.BILINEAR,
        }
        if s in mapping:
            return mapping[s]

        # tolerant fallback for strings containing keywords
        if "bicubic" in s:
            return InterpolationMode.BICUBIC
        if "bilinear" in s:
            return InterpolationMode.BILINEAR
        if "nearest" in s:
            return InterpolationMode.NEAREST
        if "lanczos" in s:
            return InterpolationMode.LANCZOS

    # final fallback
    return InterpolationMode.BICUBIC


class Rearrange:
    def __init__(self, pattern: str, **kwargs):
        self.pattern = pattern
        self.kwargs = kwargs

    def __call__(self, x):
        return rearrange(x, self.pattern, **self.kwargs)

class DivisibleCrop:
    def __init__(self, factor):
        if not isinstance(factor, tuple):
            factor = (factor, factor)

        self.height_factor, self.width_factor = factor[0], factor[1]

    def __call__(self, image: Union[torch.Tensor, Image.Image]):
        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        elif isinstance(image, Image.Image):
            width, height = image.size
        else:
            raise NotImplementedError

        cropped_height = height - (height % self.height_factor)
        cropped_width = width - (width % self.width_factor)

        image = TVF.center_crop(img=image, output_size=(cropped_height, cropped_width))
        return image


class AreaResize:
    def __init__(
        self,
        max_area: float,
        downsample_only: bool = False,
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    ):
        self.max_area = max_area
        self.downsample_only = downsample_only
        self.interpolation = interpolation

    def __call__(self, image: Union[torch.Tensor, Image.Image, List[Image.Image]]):

        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        elif isinstance(image, Image.Image):
            width, height = image.size
        elif isinstance(image, list) and isinstance(image[0], Image.Image):
            width, height = image[0].size
        else:
            raise NotImplementedError

        scale = math.sqrt(self.max_area / (height * width))

        # keep original height and width for small pictures.
        scale = 1 if scale >= 1 and self.downsample_only else scale

        resized_height, resized_width = round(height * scale), round(width * scale)



        # normalize interpolation for torchvision compatibility
        print("[NaResize] interpolation=", self.interpolation, type(self.interpolation))

        interp = _normalize_interpolation(self.interpolation)

        if isinstance(image, list) and isinstance(image[0], Image.Image):
            image = torch.stack(
                [
                    to_tensor(
                        TVF.resize(
                            _image,
                            size=(resized_height, resized_width),
                            interpolation=interp,
                        )
                    )
                    for _image in image
                ]
            )
        else:
            image = TVF.resize(
                image,
                size=(resized_height, resized_width),
                interpolation=interp,
            )
            if isinstance(image, Image.Image):
                image = to_tensor(image)
        return image

def NaResize(
    resolution, # int or list
    downsample_only: bool,
    interpolation: InterpolationMode = InterpolationMode.BICUBIC,
):

    return AreaResize(
        max_area=resolution**2,
        downsample_only=downsample_only,
        interpolation=interpolation,
    )

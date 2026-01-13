# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import numpy as np
import cv2
import time
from tqdm import tqdm
import multiprocessing
import glob

#import mediapipe as mp
#from mediapipe import solutions
#from mediapipe.framework.formats import landmark_pb2
#from mediapipe.tasks import python
#from mediapipe.tasks.python import vision
from . import face_landmark

CUR_DIR = os.path.dirname(__file__)


class LMKExtractor():
    def __init__(self, FPS=25):
        try:
            import mediapipe as mp
            from mediapipe import solutions
            from mediapipe.framework.formats import landmark_pb2
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except Exception as e:
            raise RuntimeError(
                "[DreamID-V] MediaPipe is not installed or incompatible.\n"
                "Python 3.13 is not supported for MediaPipe backend.\n"
                "If you are on Python <= 3.12, install: pip install \"mediapipe<0.10.30\".\n"
                f"Original error: {repr(e)}"
            ) from e

        self.mp = mp
        self.vision = vision
        self.mp_python = mp_python

        # Create an FaceLandmarker object.
        self.mode = self.vision.RunningMode.IMAGE
        base_options = self.mp_python.BaseOptions(
            model_asset_path=os.path.join(CUR_DIR, 'mp_models/face_landmarker_v2_with_blendshapes.task')
        )
        base_options.delegate = self.mp.tasks.BaseOptions.Delegate.CPU
        options = self.vision.FaceLandmarkerOptions(base_options=base_options,
                                            running_mode=self.mode,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
        self.detector = face_landmark.FaceLandmarker.create_from_options(options)
        self.last_ts = 0
        self.frame_ms = int(1000 / FPS)

        det_base_options = self.mp_python.BaseOptions(
            model_asset_path=os.path.join(CUR_DIR, 'mp_models/blaze_face_short_range.tflite')
        )
        det_options = self.vision.FaceDetectorOptions(base_options=det_base_options)

        self.det_detector = self.vision.FaceDetector.create_from_options(det_options)

                

    def __call__(self, img):
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame)

        t0 = time.time()
        if self.mode == self.vision.RunningMode.VIDEO:

            det_result = self.det_detector.detect(image)
            if len(det_result.detections) != 1:
                return None
            self.last_ts += self.frame_ms
            try:
                detection_result, mesh3d = self.detector.detect_for_video(image, timestamp_ms=self.last_ts)
            except Exception:
                return None
        elif self.mode == self.vision.RunningMode.IMAGE:
            # det_result = self.det_detector.detect(image)

            # if len(det_result.detections) != 1:
            #     return None
            try:
                detection_result, mesh3d = self.detector.detect(image)
            except Exception:
                return None
            
        
        bs_list = detection_result.face_blendshapes
        if len(bs_list) == 1:
            bs = bs_list[0]
            bs_values = []
            for index in range(len(bs)):
                bs_values.append(bs[index].score)
            bs_values = bs_values[1:] # remove neutral
            trans_mat = detection_result.facial_transformation_matrixes[0]
            face_landmarks_list = detection_result.face_landmarks
            face_landmarks = face_landmarks_list[0]
            lmks = []
            for index in range(len(face_landmarks)):
                x = face_landmarks[index].x
                y = face_landmarks[index].y
                z = face_landmarks[index].z
                lmks.append([x, y, z])
            lmks = np.array(lmks)
            
            lmks3d = np.array(mesh3d.vertex_buffer)
            lmks3d = lmks3d.reshape(-1, 5)[:, :3]
            mp_tris = np.array(mesh3d.index_buffer).reshape(-1, 3) + 1

            return {
                "lmks": lmks,
                'lmks3d': lmks3d,
                "trans_mat": trans_mat,
                'faces': mp_tris,
                "bs": bs_values
            }
        else:
            # print('multiple faces in the image: {}'.format(img_path))
            return None
        
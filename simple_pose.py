# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tflite_runtime.interpreter import Interpreter
import os
import numpy as np
from PIL import Image
from PIL import ImageDraw
from pose_engine import PoseEngine


pil_image = Image.open('test_data/couple.jpg').convert('RGB')
engine = PoseEngine(
    'models/mobilenet/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')
poses, inference_time = engine.DetectPosesInImage(pil_image)
print('Inference time: %.f ms' % (inference_time*1000))

draw = ImageDraw.Draw(pil_image)
for pose in poses:
  if pose.score < 0.4:
    continue
  print('\nPose Score: ', pose.score)
  for label, keypoint in pose.keypoints.items():
    print('  %-20s x=%-4d y=%-4d score=%.1f' %
          (label, keypoint.point[0], keypoint.point[1], keypoint.score))
    if keypoint.score > 0.5:
      x, y = keypoint.point
      r = 3
      draw.ellipse((x-r, y-r, x+r, y+r), fill=(0, 255, 0, 0))

pil_image.show()

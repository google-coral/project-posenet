# Lint as: python3
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


from pose_engine import PoseEngine, KeypointType
from PIL import Image
from PIL import ImageDraw
import numpy as np
import unittest
import sys
import os

PROJECT_SOURCE_DIR = os.getcwd()
sys.path.append(PROJECT_SOURCE_DIR)
MODEL_DIR = os.path.join(PROJECT_SOURCE_DIR, 'models')
EDGETPU_SHARED_LIB = 'libedgetpu.so.1'
POSENET_SHARED_LIB = os.path.join(
    PROJECT_SOURCE_DIR, 'posenet_lib', os.uname().machine, 'posenet_decoder.so')


def generate_models():
  for path, subdirs, files in os.walk(MODEL_DIR):
    for name in files:
      model_path = os.path.join(path, name)
      if 'edgetpu' in model_path:
        yield model_path


def get_random_inputs(input_shape):
  return np.random.random(input_shape).astype(np.uint8)


class PoseEngineAccuracyTest(unittest.TestCase):

  def test_runinference_all_models(self):
    for model_path in generate_models():
      print('Testing inference for:', model_path)
      engine = PoseEngine(model_path)
      image_shape = engine.get_input_tensor_shape()[1:]
      fake_image = Image.fromarray(get_random_inputs(image_shape))
      engine.DetectPosesInImage(fake_image)

  def test_model_accuracy(self):
    test_image = os.path.join(PROJECT_SOURCE_DIR, 'test_data/couple.jpg')
    image = Image.open(test_image).convert('RGB')
    for model_path in generate_models():
      print('Testing Accuracy for: ', model_path)
      engine = PoseEngine(model_path)
      poses, _ = engine.DetectPosesInImage(image)
      input_shape = engine.get_input_tensor_shape()[1:3]
      resized_image = image.resize(
          (input_shape[1], input_shape[0]), Image.NEAREST)
      draw = ImageDraw.Draw(resized_image)
      for pose in poses:
        if pose.score > 0.1:
          for label, keypoint in pose.keypoints.items():
            if keypoint.score > 0.5:
              x, y = keypoint.point
              draw.ellipse((x-3, y-3, x+3, y+3), fill=(0, 255, 0, 0))
      resized_image.show()


def main():
  unittest.main()


main()

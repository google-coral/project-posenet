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

from pycoral.utils import edgetpu
from PIL import Image
from tflite_runtime.interpreter import load_delegate
from tflite_runtime.interpreter import Interpreter

import collections
import enum
import math
import numpy as np
import os
import platform
import sys
import time


#TODO: Adds support for window and MAC
EDGETPU_SHARED_LIB = 'libedgetpu.so.1'
POSENET_SHARED_LIB = os.path.join(
    'posenet_lib', os.uname().machine, 'posenet_decoder.so')


class KeypointType(enum.IntEnum):
    """Pose kepoints."""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


Point = collections.namedtuple('Point', ['x', 'y'])
Point.distance = lambda a, b: math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)
Point.distance = staticmethod(Point.distance)

Keypoint = collections.namedtuple('Keypoint', ['point', 'score'])

Pose = collections.namedtuple('Pose', ['keypoints', 'score'])


class PoseEngine():
    """Engine used for pose tasks."""

    def __init__(self, model_path, mirror=False):
        """Creates a PoseEngine with given model.

        Args:
          model_path: String, path to TF-Lite Flatbuffer file.
          mirror: Flip keypoints horizontally.

        Raises:
          ValueError: An error occurred when model output is invalid.
        """
        edgetpu_delegate = load_delegate(EDGETPU_SHARED_LIB)
        posenet_decoder_delegate = load_delegate(POSENET_SHARED_LIB)
        self._interpreter = Interpreter(
            model_path, experimental_delegates=[edgetpu_delegate, posenet_decoder_delegate])
        self._interpreter.allocate_tensors()

        self._mirror = mirror

        self._input_tensor_shape = self.get_input_tensor_shape()
        if (self._input_tensor_shape.size != 4 or
                self._input_tensor_shape[3] != 3 or
                self._input_tensor_shape[0] != 1):
            raise ValueError(
                ('Image model should have input shape [1, height, width, 3]!'
                 ' This model has {}.'.format(self._input_tensor_shape)))
        _, self._input_height, self._input_width, self._input_depth = self.get_input_tensor_shape()
        self._input_type = self._interpreter.get_input_details()[0]['dtype']
        self._inf_time = 0

    def run_inference(self, input_data):
        """Run inference using the zero copy feature from pycoral and returns inference time in ms.
        """
        start = time.monotonic()
        edgetpu.run_inference(self._interpreter, input_data)
        self._inf_time = time.monotonic() - start
        return (self._inf_time * 1000)

    def DetectPosesInImage(self, img):
        """Detects poses in a given image.

           For ideal results make sure the image fed to this function is close to the
           expected input size - it is the caller's responsibility to resize the
           image accordingly.

        Args:
          img: numpy array containing image
        """
        input_details = self._interpreter.get_input_details()
        image_width, image_height = img.size
        resized_image = img.resize(
            (self._input_width, self._input_height), Image.NEAREST)
        input_data = np.expand_dims(resized_image, axis=0)
        if self._input_type is np.float32:
            # Floating point versions of posenet take image data in [-1,1] range.
            input_data = np.float32(resized_image) / 128.0 - 1.0
        else:
            # Assuming to be uint8
            input_data = np.asarray(resized_image)
        self.run_inference(input_data.flatten())
        return self.ParseOutput()

    def get_input_tensor_shape(self):
        """Returns input tensor shape."""
        return self._interpreter.get_input_details()[0]['shape']

    def get_output_tensor(self, idx):
        """Returns output tensor view."""
        return np.squeeze(self._interpreter.tensor(
            self._interpreter.get_output_details()[idx]['index'])())

    def ParseOutput(self):
        """Parses interpreter output tensors and returns decoded poses."""
        keypoints = self.get_output_tensor(0)
        keypoint_scores = self.get_output_tensor(1)
        pose_scores = self.get_output_tensor(2)
        num_poses = self.get_output_tensor(3)
        poses = []
        for i in range(int(num_poses)):
            pose_score = pose_scores[i]
            pose_keypoints = {}
            for j, point in enumerate(keypoints[i]):
                y, x = point
                if self._mirror:
                    y = self._input_width - y
                pose_keypoints[KeypointType(j)] = Keypoint(
                    Point(x, y), keypoint_scores[i, j])
            poses.append(Pose(pose_keypoints, pose_score))
        return poses, self._inf_time

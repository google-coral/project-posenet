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
import os
import sys
import unittest
import test_utils

test_image = os.path.join(os.getcwd(), 'test_data/test_couple.jpg')


class PoseEngineAccuracyTest(unittest.TestCase):

    def test_model_accuracy(self):
        image = Image.open(test_image).convert('RGB')
        for model_path, model_name in test_utils.generate_models():
            print('Testing Accuracy for: ', model_path)
            engine = PoseEngine(model_path)
            model_pose_result, _ = engine.DetectPosesInImage(image)
            reference_pose_scores, reference_keypoints = test_utils.parse_reference_results(
                model_name)
            score_delta = 0.1  # Allows score to change within 1 decimal place.
            pixel_delta = 4.0  # Allows placement changes of 4 pixels.
            pose_idx = 0
            keypoint_idx = 0
            for model_pose in model_pose_result:
                self.assertAlmostEqual(model_pose.score,
                                       reference_pose_scores[pose_idx], delta=score_delta)
                for _, model_keypoint in model_pose.keypoints.items():
                    reference_keypoint = reference_keypoints[keypoint_idx]
                    self.assertAlmostEqual(model_keypoint.score,
                                           reference_keypoint.score, delta=score_delta)
                    self.assertAlmostEqual(
                        model_keypoint.point[0], reference_keypoint.point[0], delta=pixel_delta)
                    self.assertAlmostEqual(
                        model_keypoint.point[1], reference_keypoint.point[1], delta=pixel_delta)
                    keypoint_idx += 1
                pose_idx += 1


def test_main():
    unittest.main()


if __name__ == '__main__':
    test_main()

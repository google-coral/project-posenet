# Lint as: python3
# Copyright 2020 Google LLC
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
"""Utilities for visualizing posenet results."""

from pose_engine import PoseEngine, Pose, Keypoint, Point
from PIL import Image
from PIL import ImageDraw

import argparse
import csv
import numpy as np
import os
import sys

PROJECT_SOURCE_DIR = os.getcwd()
sys.path.append(PROJECT_SOURCE_DIR)
MODEL_DIR = os.path.join(PROJECT_SOURCE_DIR, 'models')
EDGETPU_SHARED_LIB = 'libedgetpu.so.1'
POSENET_SHARED_LIB = os.path.join(
    PROJECT_SOURCE_DIR, 'posenet_lib', os.uname().machine, 'posenet_decoder.so')
TEST_DATA_DIR = os.path.join(PROJECT_SOURCE_DIR, 'test_data')
TEST_IMAGE = os.path.join(PROJECT_SOURCE_DIR, 'test_data/test_couple.jpg')


def generate_models():
    """Returns posenet models from MODEL_DIR."""
    for path, subdirs, files in os.walk(MODEL_DIR):
        for name in files:
            if 'component' not in path:
                model_path = os.path.join(path, name)
                yield model_path, name


def get_random_inputs(input_shape):
    """Get random input for model with size input_shape."""
    return np.random.random(input_shape).astype(np.uint8)


def write_to_csv(model_name, poses):
    """Write results of posenet model to a corresponding csv file.
    Args:
      model_name: The name of the model.
      poses: The results of the model.
    """
    csv_file_name = os.path.join(
        TEST_DATA_DIR, model_name.split('.')[0] + '_reference.csv')
    print('Writing results to:', csv_file_name)
    with open(csv_file_name, 'w') as csv_file:
        header = ['pose_id', 'pose_score', 'keypoint_label',
                  'keypoint_score', 'keypoint_x', 'keypoint_y']
        writer = csv.DictWriter(csv_file, delimiter=',', fieldnames=header)
        writer.writeheader()
        pose_id = 0
        line_dict = {}
        for pose in poses:
            line_dict['pose_id'] = pose_id
            line_dict['pose_score'] = pose.score
            for label, keypoint in pose.keypoints.items():
                line_dict['keypoint_label'] = label.name
                line_dict['keypoint_score'] = keypoint.score
                line_dict['keypoint_x'] = keypoint.point[0]
                line_dict['keypoint_y'] = keypoint.point[1]
                writer.writerow(line_dict)
            pose_id += 1


def visualize_results_from_model(model_name, image, input_shape, poses):
    """Visualize inference results from a model.
    Args:
      model_name: the name of the model.
      image: the image that the result was from.
      input_shape: the model's input shape.
      poses: the results from the model.
    """
    print('Visualizing model results for:', model_name)
    resized_image = image.resize(
        (input_shape[1], input_shape[0]), Image.NEAREST)
    draw = ImageDraw.Draw(resized_image)
    draw.text((10, 5), model_name, fill=(0, 255, 0))
    for pose in poses:
        if pose.score > 0.5:
            for label, keypoint in pose.keypoints.items():
                if keypoint.score > 0.5:
                    x, y = keypoint.point
                    draw.ellipse((x-2, y-2, x+2, y+2), fill=(0, 255, 0, 0))
    resized_image.show()


def visualize_results_from_reference_file(model_name, image, input_shape):
    """Visualize the reference result from a model.
    Args:
      model_name: the name of the model.
      image: the image that the result was from.
      input_shape: the model's input shape.
    """
    resized_image = image.resize(
        (input_shape[1], input_shape[0]), Image.NEAREST)
    draw = ImageDraw.Draw(resized_image)
    csv_file_name = os.path.join(
        TEST_DATA_DIR, model_name.split('.')[0] + '_reference.csv')
    print('Visualizing reference results for:', csv_file_name)
    draw.text((10, 5), csv_file_name.split('/')[-1], fill=(255, 0, 0))
    with open(csv_file_name, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            x = float(row['keypoint_x'])
            y = float(row['keypoint_y'])
            draw.ellipse((x-2, y-2, x+2, y+2), fill=(255, 0, 0, 0))
    resized_image.show()


def parse_reference_results(model_name):
    """Parse reference results for the given model and return a list of keypoints."""
    csv_file_name = os.path.join(
        TEST_DATA_DIR, model_name.split('.')[0] + '_reference.csv')
    print('Parsing reference results for:', csv_file_name)
    keypoints = []
    pose_scores = [0.0, 0.0]
    with open(csv_file_name, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            pose_scores[int(row['pose_id'])] = float(row['pose_score'])
            keypoint_score = float(row['keypoint_score'])
            keypoint_x = float(row['keypoint_x'])
            keypoint_y = float(row['keypoint_y'])
            keypoints.append(
                Keypoint(Point(keypoint_x, keypoint_y), keypoint_score))
    return pose_scores, keypoints


def generate_results(write_csv=False, visualize_model_results=False, visualize_reference_results=False):
    """Generates results form a model (both from reference results or from new inference).
    Args:
      write_csv: Whether to write new inference results to a csv file or not.
      visualize_model_results: Whether to visualize the new results or not.
      visualize_reference_result: Whether to visualize the reference results or not.
    """
    image = Image.open(TEST_IMAGE).convert('RGB')
    for model_path, model_name in generate_models():
        engine = PoseEngine(model_path)
        poses, _ = engine.DetectPosesInImage(image)
        if write_csv:
            write_to_csv(model_name, poses)
        input_shape = engine.get_input_tensor_shape()[1:3]
        if visualize_model_results:
            visualize_results_from_model(model_name, image, input_shape, poses)
        if visualize_reference_results:
            visualize_results_from_reference_file(
                model_name, image, input_shape)


def test_utils_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write_csv', default=False,
                        action='store_true', help='Write new reference result to csv.')
    parser.add_argument('--visualize_model_results',
                        action='store_true', default=False, help='Visualize new model results.')
    parser.add_argument('--visualize_reference_results',
                        action='store_true', default=False, help='Visualize old reference result from csv.')
    args = parser.parse_args()
    generate_results(args.write_csv, args.visualize_model_results,
                     args.visualize_reference_results)


if __name__ == '__main__':
    test_utils_main()

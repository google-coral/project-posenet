from pose_engine import PoseEngine, KeypointType
from PIL import Image
from PIL import ImageDraw
import numpy as np
import unittest
import sys
import os
import csv
import argparse

PROJECT_SOURCE_DIR = os.getcwd()
sys.path.append(PROJECT_SOURCE_DIR)
MODEL_DIR = os.path.join(PROJECT_SOURCE_DIR, 'models')
EDGETPU_SHARED_LIB = 'libedgetpu.so.1'
POSENET_SHARED_LIB = os.path.join(
    PROJECT_SOURCE_DIR, 'posenet_lib', os.uname().machine, 'posenet_decoder.so')
TEST_DATA_DIR = os.path.join(PROJECT_SOURCE_DIR, 'test_data')
TEST_IMAGE = os.path.join(PROJECT_SOURCE_DIR, 'test_data/test_couple.jpg')


def generate_models():
  for path, subdirs, files in os.walk(MODEL_DIR):
    for name in files:
      if 'component' not in path:
        model_path = os.path.join(path, name)
        yield model_path, name


def write_to_csv(model_name, poses):
  csv_file_name = os.path.join(
      TEST_DATA_DIR, model_name.split('.')[0] + '_reference.csv')
  print('Writing results to:', csv_file_name)
  with open(csv_file_name, 'w') as csv_file:
    header = ['pose_id', 'keypoint_label',
              'keypoint_score', 'keypoint_x', 'keypoint_y']
    writer = csv.DictWriter(csv_file, delimiter=',', fieldnames=header)
    writer.writeheader()
    pose_id = 0
    line_dict = {}
    for pose in poses:
      line_dict['pose_id'] = pose_id
      for label, keypoint in pose.keypoints.items():
        line_dict['keypoint_label'] = label
        line_dict['keypoint_score'] = keypoint.score
        line_dict['keypoint_x'] = keypoint.point[0]
        line_dict['keypoint_y'] = keypoint.point[1]
        writer.writerow(line_dict)
      pose_id += 1


def visualize_results_from_model(model_name, image, input_shape, poses):
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


def visualize_reference_results(model_name, image, input_shape):
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


def generate_results(write_csv=False, visualize_model_results=False, visualize_reference_result=False):
  image = Image.open(TEST_IMAGE).convert('RGB')
  for model_path, model_name in generate_models():
    engine = PoseEngine(model_path)
    poses, _ = engine.DetectPosesInImage(image)
    input_shape = engine.get_input_tensor_shape()[1:3]
    if write_csv:
      write_to_csv(model_name, poses)
    if visualize_model_results:
      visualize_results_from_model(model_name, image, input_shape, poses)
    if visualize_reference_result:
      visualize_reference_results(model_name, image, input_shape)


def test_utils_main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--write_csv', type=bool, default=False,
                      required=False, help='Write new reference result to csv.')
  parser.add_argument('--visualize_model_results', type=bool,
                      required=False, default=True, help='Visualize new model results.')
  parser.add_argument('--visualize_reference_results', type=bool,
                      required=False, default=True, help='Visualize old reference result from csv.')
  args = parser.parse_args()
  generate_results(args.write_csv, args.visualize_model_results,
                   args.visualize_reference_results)


test_utils_main()

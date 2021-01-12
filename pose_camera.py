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

import argparse
import collections
from functools import partial
import re
import time

import numpy as np
from PIL import Image
import svgwrite
import gstreamer

from pose_engine import PoseEngine
from pose_engine import KeypointType

EDGES = (
    (KeypointType.NOSE, KeypointType.LEFT_EYE),
    (KeypointType.NOSE, KeypointType.RIGHT_EYE),
    (KeypointType.NOSE, KeypointType.LEFT_EAR),
    (KeypointType.NOSE, KeypointType.RIGHT_EAR),
    (KeypointType.LEFT_EAR, KeypointType.LEFT_EYE),
    (KeypointType.RIGHT_EAR, KeypointType.RIGHT_EYE),
    (KeypointType.LEFT_EYE, KeypointType.RIGHT_EYE),
    (KeypointType.LEFT_SHOULDER, KeypointType.RIGHT_SHOULDER),
    (KeypointType.LEFT_SHOULDER, KeypointType.LEFT_ELBOW),
    (KeypointType.LEFT_SHOULDER, KeypointType.LEFT_HIP),
    (KeypointType.RIGHT_SHOULDER, KeypointType.RIGHT_ELBOW),
    (KeypointType.RIGHT_SHOULDER, KeypointType.RIGHT_HIP),
    (KeypointType.LEFT_ELBOW, KeypointType.LEFT_WRIST),
    (KeypointType.RIGHT_ELBOW, KeypointType.RIGHT_WRIST),
    (KeypointType.LEFT_HIP, KeypointType.RIGHT_HIP),
    (KeypointType.LEFT_HIP, KeypointType.LEFT_KNEE),
    (KeypointType.RIGHT_HIP, KeypointType.RIGHT_KNEE),
    (KeypointType.LEFT_KNEE, KeypointType.LEFT_ANKLE),
    (KeypointType.RIGHT_KNEE, KeypointType.RIGHT_ANKLE),
)


def shadow_text(dwg, x, y, text, font_size=16):
    dwg.add(dwg.text(text, insert=(x + 1, y + 1), fill='black',
                     font_size=font_size, style='font-family:sans-serif'))
    dwg.add(dwg.text(text, insert=(x, y), fill='white',
                     font_size=font_size, style='font-family:sans-serif'))


def draw_pose(dwg, pose, src_size, inference_box, color='yellow', threshold=0.2):
    box_x, box_y, box_w, box_h = inference_box
    scale_x, scale_y = src_size[0] / box_w, src_size[1] / box_h
    xys = {}
    for label, keypoint in pose.keypoints.items():
        if keypoint.score < threshold: continue
        # Offset and scale to source coordinate space.
        kp_x = int((keypoint.point[0] - box_x) * scale_x)
        kp_y = int((keypoint.point[1] - box_y) * scale_y)

        xys[label] = (kp_x, kp_y)
        dwg.add(dwg.circle(center=(int(kp_x), int(kp_y)), r=5,
                           fill='cyan', fill_opacity=keypoint.score, stroke=color))

    for a, b in EDGES:
        if a not in xys or b not in xys: continue
        ax, ay = xys[a]
        bx, by = xys[b]
        dwg.add(dwg.line(start=(ax, ay), end=(bx, by), stroke=color, stroke_width=2))


def avg_fps_counter(window_size):
    window = collections.deque(maxlen=window_size)
    prev = time.monotonic()
    yield 0.0  # First fps value.

    while True:
        curr = time.monotonic()
        window.append(curr - prev)
        prev = curr
        yield len(window) / sum(window)


def run(inf_callback, render_callback):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mirror', help='flip video horizontally', action='store_true')
    parser.add_argument('--model', help='.tflite model path.', required=False)
    parser.add_argument('--res', help='Resolution', default='640x480',
                        choices=['480x360', '640x480', '1280x720'])
    parser.add_argument('--videosrc', help='Which video source to use', default='/dev/video0')
    parser.add_argument('--h264', help='Use video/x-h264 input', action='store_true')
    parser.add_argument('--jpeg', help='Use image/jpeg input', action='store_true')
    args = parser.parse_args()

    default_model = 'models/mobilenet/posenet_mobilenet_v1_075_%d_%d_quant_decoder_edgetpu.tflite'
    if args.res == '480x360':
        src_size = (640, 480)
        appsink_size = (480, 360)
        model = args.model or default_model % (353, 481)
    elif args.res == '640x480':
        src_size = (640, 480)
        appsink_size = (640, 480)
        model = args.model or default_model % (481, 641)
    elif args.res == '1280x720':
        src_size = (1280, 720)
        appsink_size = (1280, 720)
        model = args.model or default_model % (721, 1281)

    print('Loading model: ', model)
    engine = PoseEngine(model)
    input_shape = engine.get_input_tensor_shape()
    inference_size = (input_shape[2], input_shape[1])

    gstreamer.run_pipeline(partial(inf_callback, engine), partial(render_callback, engine),
                           src_size, inference_size,
                           mirror=args.mirror,
                           videosrc=args.videosrc,
                           h264=args.h264,
                           jpeg=args.jpeg
                           )


def main():
    n = 0
    sum_process_time = 0
    sum_inference_time = 0
    ctr = 0
    fps_counter = avg_fps_counter(30)

    def run_inference(engine, input_tensor):
        return engine.run_inference(input_tensor)

    def render_overlay(engine, output, src_size, inference_box):
        nonlocal n, sum_process_time, sum_inference_time, fps_counter

        svg_canvas = svgwrite.Drawing('', size=src_size)
        start_time = time.monotonic()
        outputs, inference_time = engine.ParseOutput()
        end_time = time.monotonic()
        n += 1
        sum_process_time += 1000 * (end_time - start_time)
        sum_inference_time += inference_time * 1000

        avg_inference_time = sum_inference_time / n
        text_line = 'PoseNet: %.1fms (%.2f fps) TrueFPS: %.2f Nposes %d' % (
            avg_inference_time, 1000 / avg_inference_time, next(fps_counter), len(outputs)
        )

        shadow_text(svg_canvas, 10, 20, text_line)
        for pose in outputs:
            draw_pose(svg_canvas, pose, src_size, inference_box)
        return (svg_canvas.tostring(), False)

    run(run_inference, render_overlay)


if __name__ == '__main__':
    main()

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

import svgwrite
import time

import pose_camera

BACKGROUND_DELAY = 2  # seconds


def main():
    background_locked = False
    timer_time = time.monotonic()

    def run_inference(engine, input_tensor):
        return engine.run_inference(input_tensor)

    def render_overlay(engine, output, src_size, inference_box):
        nonlocal timer_time, background_locked
        svg_canvas = svgwrite.Drawing('', size=src_size)
        outputs, inference_time = engine.ParseOutput()
        now_time = time.monotonic()

        if not background_locked:
            print('Waiting for everyone to leave the frame...')
            pose_camera.shadow_text(svg_canvas, 10, 20,
                                    'Waiting for everyone to leave the frame...')
            if outputs:  # frame still has people in it, restart timer
                timer_time = now_time
            elif now_time > timer_time + BACKGROUND_DELAY:  # frame has been empty long enough
                background_locked = True
                print('Background set.')

        for pose in outputs:
            pose_camera.draw_pose(svg_canvas, pose, src_size, inference_box)

        return (svg_canvas.tostring(), background_locked)

    pose_camera.run(run_inference, render_overlay)


if __name__ == '__main__':
    main()

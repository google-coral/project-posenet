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

import time

import pose_camera

BACKGROUND_DELAY = 2  # seconds


def main():
    background_image = None
    timer_time = time.monotonic()

    def render_overlay(engine, image, svg_canvas):
        nonlocal timer_time, background_image
        outputs, inference_time = engine.DetectPosesInImage(image)
        now_time = time.monotonic()

        if background_image is None:
            pose_camera.shadow_text(svg_canvas, 10, 20,
                                    'Waiting for everyone to leave the frame...')
            if outputs:  # frame still has people in it, restart timer
                print('Waiting for everyone to leave the frame...')
                timer_time = now_time
            elif now_time > timer_time + BACKGROUND_DELAY:  # frame has been empty long enough
                background_image = image
                print('Background set.')
        else:
            image = background_image

        for pose in outputs:
            pose_camera.draw_pose(svg_canvas, pose)

        return image

    pose_camera.run(render_overlay, use_appsrc=True)


if __name__ == '__main__':
    main()

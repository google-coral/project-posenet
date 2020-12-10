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

import itertools
import svgwrite
import time

import numpy as np
import fluidsynth

import pose_camera

OCTAVE = 12
FIFTH = 7
MINOR_THIRD = 3
CIRCLE_OF_FIFTHS = tuple((i * FIFTH) % OCTAVE
                         for i in range(OCTAVE))

# first 5 fifths in order, e.g. C G D A E => C D E G A
MAJOR_PENTATONIC = tuple(sorted(CIRCLE_OF_FIFTHS[:5]))

# same as pentatonic major but starting at 9
# e.g. C D E G A = 0 2 4 7 9 => 3 5 7 10 0 = C D E G A => A C D E G
MINOR_PENTATONIC = tuple(sorted((i + MINOR_THIRD) % OCTAVE
                                for i in MAJOR_PENTATONIC))

SCALE = MAJOR_PENTATONIC

# General Midi ids
OVERDRIVEN_GUITAR = 30
ELECTRIC_BASS_FINGER = 34
VOICE_OOHS = 54

CHANNELS = (OVERDRIVEN_GUITAR, ELECTRIC_BASS_FINGER, VOICE_OOHS)


class Identity:
    def __init__(self, color, base_note, instrument, extent=2*OCTAVE):
        self.color = color
        self.base_note = base_note
        self.channel = CHANNELS.index(instrument)
        self.extent = extent


IDENTITIES = (
    Identity('cyan', 24, OVERDRIVEN_GUITAR),
    Identity('magenta', 12, ELECTRIC_BASS_FINGER),
    Identity('yellow', 36, VOICE_OOHS),
)


class Pose:
    def __init__(self, pose, threshold):
        self.pose = pose
        self.id = None
        self.keypoints = {label: k for label, k in pose.keypoints.items()
                          if k.score > threshold}
        self.center = (np.mean([k.point for k in self.keypoints.values()], axis=0)
                       if self.keypoints else None)

    def quadrance(self, other):
        d = self.center - other.center
        return d.dot(d)


class PoseTracker:
    def __init__(self):
        self.prev_poses = []
        self.next_pose_id = 0

    def assign_pose_ids(self, poses):
        """copy nearest pose ids from previous frame to current frame"""
        all_pairs = sorted(itertools.product(poses, self.prev_poses),
                           key=lambda pair: pair[0].quadrance(pair[1]))
        used_ids = set()
        for pose, prev_pose in all_pairs:
            if pose.id is None and prev_pose.id not in used_ids:
                pose.id = prev_pose.id
                used_ids.add(pose.id)

        for pose in poses:
            if pose.id is None:
                pose.id = self.next_pose_id
                self.next_pose_id += 1

        self.prev_poses = poses


def main():
    pose_tracker = PoseTracker()
    synth = fluidsynth.Synth()

    synth.start('alsa')
    soundfont_id = synth.sfload('/usr/share/sounds/sf2/FluidR3_GM.sf2')
    for channel, instrument in enumerate(CHANNELS):
        synth.program_select(channel, soundfont_id, 0, instrument)

    prev_notes = set()

    def run_inference(engine, input_tensor):
        return engine.run_inference(input_tensor)

    def render_overlay(engine, output, src_size, inference_box):
        nonlocal prev_notes
        svg_canvas = svgwrite.Drawing('', size=src_size)
        outputs, inference_time = engine.ParseOutput()

        poses = [pose for pose in (Pose(pose, 0.2) for pose in outputs)
                 if pose.keypoints]
        pose_tracker.assign_pose_ids(poses)

        velocities = {}
        for pose in poses:
            left = pose.keypoints.get('left wrist')
            right = pose.keypoints.get('right wrist')
            if not (left and right): continue

            identity = IDENTITIES[pose.id % len(IDENTITIES)]
            left = 1 - left.point[0] / engine.image_height
            right = 1 - right.point[0] / engine.image_height
            velocity = int(left * 100)
            i = int(right * identity.extent)
            note = (identity.base_note
                    + OCTAVE * (i // len(SCALE))
                    + SCALE[i % len(SCALE)])
            velocities[(identity.channel, note)] = velocity

        for note in prev_notes:
            if note not in velocities: synth.noteoff(*note)
        for note, velocity in velocities.items():
            if note not in prev_notes:
                synth.noteon(*note, velocity)
        prev_notes = velocities.keys()

        for i, pose in enumerate(poses):
            identity = IDENTITIES[pose.id % len(IDENTITIES)]
            pose_camera.draw_pose(svg_canvas, pose, src_size, inference_box, color=identity.color)

        return (svg_canvas.tostring(), False)

    pose_camera.run(run_inference, render_overlay)


if __name__ == '__main__':
    main()

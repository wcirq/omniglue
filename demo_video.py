#!/usr/bin/env python3
# Copyright 2024 Google LLC
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

"""Demo script for performing OmniGlue inference."""
import cv2
import numpy as np
import omniglue
from omniglue import utils


def main() -> None:

    og = omniglue.OmniGlue(
        og_export="./models/og_export",
        # sp_export="./models/sp_v6",
        sp_export="./models/sp_torch/superpoint_v1.pth",
        dino_export="./models/dinov2_vitb14_pretrain.pth",
    )

    cap = cv2.VideoCapture(0)

    fxy = 0.5
    image0 = None
    for i in range(100):
        ret, image = cap.read()
        image = cv2.resize(image, (0, 0), fx=fxy, fy=fxy)
        image0 = image
        cv2.waitKey(10)

    while True:
        ret, image = cap.read()
        if not ret:
            break

        if image0 is None:
            break

        image = cv2.resize(image, (0, 0), fx=fxy, fy=fxy)
        image1 = image
        match_kp0, match_kp1, match_confidences = og.FindMatches(image0, image1)

        # Filter by confidence (0.02).
        print("> Filtering matches...")
        match_threshold = 0.02  # Choose any value [0.0, 1.0).
        keep_idx = []
        for i in range(match_kp0.shape[0]):
            if match_confidences[i] > match_threshold:
                keep_idx.append(i)
        num_filtered_matches = len(keep_idx)
        match_kp0 = match_kp0[keep_idx]
        match_kp1 = match_kp1[keep_idx]
        match_confidences = match_confidences[keep_idx]

        viz = utils.visualize_matches(
            image0,
            image1,
            match_kp0,
            match_kp1,
            np.eye(num_filtered_matches),
            show_keypoints=True,
            highlight_unmatched=True,
            title=f"{num_filtered_matches} matches",
            line_width=2,
        )

        cv2.imshow("image", viz)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()

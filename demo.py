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

import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import omniglue
from omniglue import utils
from PIL import Image


def main(argv) -> None:
    # image_path1 = "res/demo1.jpg"
    # image_path2 = "res/demo2.jpg"
    image_path1 = r"E:\fromwcirq\source\multimedia-lose-color-recognition\x64\Release\images\product\ll\20241015142921.jpg"
    image_path2 = r"E:\fromwcirq\source\multimedia-lose-color-recognition\x64\Release\images\product\ll\20241015142935.jpg"

    # Load images.
    print("> Loading images...")
    image0 = np.array(Image.open(image_path1).convert("RGB").resize((780, 582)))
    image1 = np.array(Image.open(image_path2).convert("RGB").resize((780, 582)))

    # Load models.
    print("> Loading OmniGlue (and its submodules: SuperPoint & DINOv2)...")
    start = time.time()
    og = omniglue.OmniGlue(
        og_export="./models/og_export",
        # sp_export="./models/sp_v6",
        sp_export="./models/sp_torch/superpoint_v1.pth",
        dino_export="./models/dinov2_vitb14_pretrain.pth",
    )
    print(f"> \tTook {time.time() - start} seconds.")

    # Perform inference.
    print("> Finding matches...")
    start = time.time()
    match_kp0, match_kp1, match_confidences = og.FindMatches(image0, image1)
    num_matches = match_kp0.shape[0]
    print(f"> \tFound {num_matches} matches.")
    print(f"> \tTook {time.time() - start} seconds.")

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
    print(f"> \tFound {num_filtered_matches}/{num_matches} above threshold {match_threshold}")

    # match_kp0, match_kp1, match_confidences = match_kp0[::3], match_kp1[::3], match_confidences[::3]

    # Visualize.
    print("> Visualizing matches...")
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
    plt.figure(figsize=(20, 10), dpi=100, facecolor="w", edgecolor="k")
    plt.axis("off")
    plt.imshow(viz)
    plt.imsave("./demo_output.png", viz)
    print("> \tSaved visualization to ./demo_output.png")


if __name__ == "__main__":
    main(sys.argv)

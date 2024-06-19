# Artificial Senses
# Copyright (C) 2024 Archimedes Exhibitions GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import cv2
from ultralytics import YOLO


class Segmentation:
    def __init__(self, include_labels):
        self._include_labels = include_labels
        self._yolo_model = YOLO("yolov8n-seg.pt")

    def process(self, color_image, depth_frame):
        results = self._yolo_model.predict(color_image, stream=True, verbose=False)
        segmented_image = np.zeros(color_image.shape, np.uint8)
        centroids = []

        if results:
            result = list(results)[0]

            all_contours = []
            for ci, c in enumerate(result):
                label = c.names[c.boxes.cls.tolist().pop()]
                if label not in self._include_labels:
                    continue

                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                _ = cv2.drawContours(
                    segmented_image, [contour], -1, (0, 0, 255), cv2.FILLED
                )
                all_contours.append(contour)

            segmented_image = cv2.addWeighted(color_image, 1, segmented_image, 0.5, 0)
            for contour in all_contours:
                moments = cv2.moments(contour)
                if moments["m00"] != 0:
                    # Calculate the centroid from moments
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])

                    centroids.append((cx, cy, depth_frame.get_distance(cx, cy)))

        return segmented_image, centroids

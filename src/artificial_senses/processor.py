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

import queue
import threading
import logging
from dataclasses import dataclass
from typing import Union

import cv2
import numpy as np
import pyglet

from artificial_senses import segmentation

logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    color_image: pyglet.image.ImageData
    depth_paletted_image: pyglet.image.ImageData
    segmented_image: pyglet.image.ImageData
    centroids: list
    pointcloud_vertexes: np.array
    pointcloud_colors: np.array


class Processor(threading.Thread):
    def __init__(self, camera):
        super().__init__(daemon=True)
        self._camera = camera
        # TODO: move to config
        self._segmentation = segmentation.Segmentation(include_labels=["person"])
        self._running = False
        self._queue = queue.Queue()

    def run(self):
        self._running = True
        logger.info("Starting processor thread")
        while self._running:
            self._process_frame()

    def stop(self):
        logger.info("Stopping processor thread")
        self._running = False
        self.join()

    def get_dataset(self) -> Union[None, Dataset]:
        if self._queue.empty():
            return None
        else:
            return self._queue.get_nowait()

    def _cv2img_to_pyglet(self, image) -> pyglet.image.ImageData:
        # Convert the frame from BGR to RGB (Pyglet uses RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 0)

        # Convert the frame to Pyglet image format
        return pyglet.image.ImageData(
            image.shape[1], image.shape[0], "RGB", image.tobytes()
        )

    def _nparray_to_cv2img(self, image) -> pyglet.image.ImageData:
        pyglet_image = pyglet.image.ImageData(
            image.shape[1],
            image.shape[0],
            "RGB",
            image.tobytes(),
        )
        return pyglet_image

    def _process_frame(self):
        frameset = self._camera.get_frames()
        segmented_image, centroids = self._segmentation.process(
            frameset.color_image, frameset.depth_frame
        )
        centroids = [
            (c[0], c[1], int(c[2] / self._camera.depth_scale)) for c in centroids
        ]

        self._queue.put(
            Dataset(
                color_image=self._cv2img_to_pyglet(frameset.color_image),
                depth_paletted_image=self._nparray_to_cv2img(frameset.depth_image),
                segmented_image=self._cv2img_to_pyglet(segmented_image),
                centroids=centroids,
                pointcloud_vertexes=frameset.pointcloud_vertexes,
                pointcloud_colors=frameset.pointcloud_colors,
            )
        )

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

import logging

from dataclasses import dataclass
import pyrealsense2 as rs
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Frameset:
    depth_frame: rs.frame
    depth_image: np.ndarray
    color_image: np.ndarray
    pointcloud_vertexes: np.ndarray
    pointcloud_colors: np.ndarray


class RealSenseCamera:
    CAPTURE_WIDTH = 640
    CAPTURE_HEIGHT = 480
    CAPTURE_FPS = 30

    def __init__(self):
        # Configure depth and color streams
        self._align = rs.align(rs.stream.color)
        self._pipeline = rs.pipeline()
        self._colorizer = rs.colorizer()
        self._pointcloud = rs.pointcloud()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self._pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        logger.info("Opened camera:")
        logger.info(f"  Device PID: {device.get_info(rs.camera_info.product_id)}")
        logger.info(f"  Device name: {device.get_info(rs.camera_info.name)}")
        logger.info(f"  Serial number: {device.get_info(rs.camera_info.serial_number)}")
        logger.info(
            f"  Firmware version: {device.get_info(rs.camera_info.firmware_version)}"
        )

        config.enable_stream(
            rs.stream.depth,
            self.CAPTURE_WIDTH,
            self.CAPTURE_HEIGHT,
            rs.format.z16,
            self.CAPTURE_FPS,
        )
        config.enable_stream(
            rs.stream.color,
            self.CAPTURE_WIDTH,
            self.CAPTURE_HEIGHT,
            rs.format.bgr8,
            self.CAPTURE_FPS,
        )

        # Start streaming
        profile = self._pipeline.start(config)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        self._depth_intrinsics = depth_profile.get_intrinsics()
        self._depth_width = self._depth_intrinsics.width
        self._depth_height = self._depth_intrinsics.height

    @property
    def depth_size(self):
        return self._depth_width, self._depth_height

    def get_frames(self):
        while True:
            frames = self._pipeline.wait_for_frames()

            aligned_frames = self._align.process(frames)

            # Get aligned frames
            depth_frame = (
                aligned_frames.get_depth_frame()
            )  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            colorized_depth = self._colorizer.colorize(depth_frame)
            ones = (
                np.ones((self._depth_height, self._depth_width, 1), dtype=np.uint8)
                * 255
            )
            depth_colormap = np.asanyarray(colorized_depth.get_data())
            colors = np.concatenate((depth_colormap, ones), axis=-1)
            points = self._pointcloud.calculate(depth_frame)
            vertexes = np.asarray(points.get_vertices(2)).reshape(
                self._depth_height, self._depth_width, 3
            )

            return Frameset(
                depth_frame=depth_frame,
                depth_image=np.flipud(depth_colormap),
                color_image=color_image,
                pointcloud_vertexes=vertexes,
                pointcloud_colors=colors,
            )

    def deproject_pixel_to_point(self, point, depth):
        return rs.rs2_deproject_pixel_to_point(self._depth_intrinsics, point, depth)

    def stop(self):
        self._pipeline.stop()

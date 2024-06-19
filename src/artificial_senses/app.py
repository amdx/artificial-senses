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

import sys
import logging
import math
from pathlib import Path

import numpy as np
import pyglet
from pyglet import gl
from pyglet.math import Mat4
from pyglet.math import Vec3

from artificial_senses import processor
from artificial_senses import camera
from artificial_senses import __version__

logger = logging.getLogger(__name__)


class AppWindow(pyglet.window.Window):
    FLYBY_THETA_INCREASE = 0.15
    REFERENCE_WIDTH = 1920
    REFERENCE_HEIGHT = 1080
    BACKGROUND_COLOR = (0x26, 0x26, 0x26, 0xFF)
    TEXT_COLOR = (0xFF, 0xFF, 0xFF, 0xFF)
    MASK_COLOR = (0xBB, 0xC3, 0x92, 0xFF)
    FRUSTRUM_LINES_COLOR = (0x7F, 0x7F, 0x7F, 0x10)

    def __init__(self):
        logger.info(f"Artificial senses {__version__}")
        logger.info(f"  pyglet version={pyglet.version}")

        try:
            self._camera = camera.RealSenseCamera()
        except RuntimeError:
            logger.error("No realsense camera found, exiting")
            sys.exit(1)

        config = gl.Config(
            sample_buffers=1, samples=4, depth_size=16, double_buffer=True
        )
        super().__init__(fullscreen=True, config=config)

        logger.info(f"  screen size={self.size}")

        self._processor = processor.Processor(self._camera)
        self._processor.start()
        self._current_dataset = None
        shader = pyglet.graphics.get_default_shader()

        self._pointcloud_batch = pyglet.graphics.Batch()
        self._pointcloud_vlist = shader.vertex_list(
            self._camera.depth_size[0] * self._camera.depth_size[1],
            gl.GL_POINTS,
            batch=self._pointcloud_batch,
            position="f",
            colors="Bn",
        )

        self._cursor_image = pyglet.image.load(
            Path(__file__).parent / "data" / "raster" / "cursor.png"
        )
        self._cursor_image.anchor_x = self._cursor_image.width // 2
        self._cursor_image.anchor_y = self._cursor_image.height // 2

        self._frustrum_batch = pyglet.graphics.Batch()
        self._compute_frustrum()

        self._mask_batch = pyglet.graphics.Batch()
        self._mask_shapes = []
        self._compute_mask()

        self._flyby_theta = 0
        pyglet.clock.schedule_interval(self._tick_flyby, 1 / 60)
        self._set_bg_color(*self.BACKGROUND_COLOR)

    def stop(self):
        self._processor.stop()
        self._camera.stop()

    def on_draw(self):
        self.clear()

        dataset = self._processor.get_dataset()
        if dataset:
            self._current_dataset = dataset

        if self._current_dataset:
            self._set_perspective()
            self._render_pointcloud(
                self._current_dataset.pointcloud_vertexes,
                self._current_dataset.pointcloud_colors,
            )
            self._reset_camera()

            yoffs = self.REFERENCE_HEIGHT - self._camera.CAPTURE_HEIGHT
            self._current_dataset.color_image.blit(0, yoffs)
            self._current_dataset.depth_paletted_image.blit(
                self._camera.CAPTURE_WIDTH, yoffs
            )
            self._current_dataset.segmented_image.blit(
                self._camera.CAPTURE_WIDTH * 2, yoffs
            )

            self._draw_centroids_labels(self._current_dataset.centroids)

            self._mask_batch.draw()
        else:
            label = pyglet.text.Label(
                "Initializing RGB/depth stream",
                font_size=24,
                x=self.width / 2,
                y=self.height / 2,
                anchor_x="center",
                anchor_y="center",
                color=self.TEXT_COLOR,
            )
            label.draw()

    def _set_bg_color(self, r, g, b, a):
        gl.glClearColor(r / 255, g / 255, b / 255, a / 255)

    def _tick_flyby(self, dt):
        self._flyby_theta += self.FLYBY_THETA_INCREASE * dt

    def _set_perspective(self):
        self.projection = Mat4.perspective_projection(
            self.aspect_ratio, z_near=0.1, z_far=255, fov=60
        )

    def _reset_camera(self):
        self.view = Mat4()
        self.projection = Mat4.orthogonal_projection(
            0, self.REFERENCE_WIDTH, 0, self.REFERENCE_HEIGHT, z_near=-255, z_far=255
        )

    def _draw_centroids_labels(self, centroids):
        centroids_batch = pyglet.graphics.Batch()
        shapes_temp = []
        cursors = []
        for cx, cy, distance in centroids:
            # Offset to the third slot's left edge
            cx += self._camera.CAPTURE_WIDTH * 2
            cy += self.REFERENCE_HEIGHT - self._camera.CAPTURE_HEIGHT

            cursors.append(
                pyglet.sprite.Sprite(self._cursor_image, cx, cy, batch=centroids_batch)
            )

            if distance > 0:
                label = pyglet.text.Label(
                    text=f"{distance}mm",
                    font_size=18,
                    x=cx + self._cursor_image.width / 2 + 2,
                    y=cy + 2,
                    anchor_x="left",
                    anchor_y="center",
                    color=self.TEXT_COLOR,
                    batch=centroids_batch,
                )

                if cx > self.REFERENCE_WIDTH - label.content_width:
                    label.x = cx - self._cursor_image.width / 2 - 2
                    label.anchor_x = "right"

                shapes_temp.append(label)

        centroids_batch.draw()

    def _compute_frustrum(self):
        coordinates = []
        for d in range(1, 6, 2):

            def get_point(x, y):
                p = self._camera.deproject_pixel_to_point((x, y), d)
                coordinates.append([0, 0, 0] + p)
                return p

            top_left = get_point(0, 0)
            top_right = get_point(self._camera.depth_size[0], 0)
            bottom_right = get_point(
                self._camera.depth_size[0], self._camera.depth_size[1]
            )
            bottom_left = get_point(0, self._camera.depth_size[1])

            coordinates.append(top_left + top_right)
            coordinates.append(top_right + bottom_right)
            coordinates.append(bottom_right + bottom_left)
            coordinates.append(bottom_left + top_left)

        shader = pyglet.graphics.get_default_shader()
        lines_coords_count = len(coordinates) * 2
        shader.vertex_list(
            lines_coords_count,
            gl.GL_LINES,
            batch=self._frustrum_batch,
            position=("f", np.ravel(coordinates)),
            colors=(
                "f",
                [c / 255 for c in self.FRUSTRUM_LINES_COLOR] * lines_coords_count,
            ),
        )

    def _compute_mask(self):
        y = self.REFERENCE_HEIGHT - self._camera.CAPTURE_HEIGHT
        self._mask_shapes.append(
            pyglet.shapes.Line(
                0,
                y,
                self.REFERENCE_WIDTH,
                y,
                color=self.MASK_COLOR,
                batch=self._mask_batch,
            )
        )
        self._mask_shapes.append(
            pyglet.shapes.Line(
                self._camera.CAPTURE_WIDTH,
                self.REFERENCE_HEIGHT,
                self._camera.CAPTURE_WIDTH,
                y,
                color=self.MASK_COLOR,
                batch=self._mask_batch,
            )
        )
        self._mask_shapes.append(
            pyglet.shapes.Line(
                self._camera.CAPTURE_WIDTH * 2,
                self.REFERENCE_HEIGHT,
                self._camera.CAPTURE_WIDTH * 2,
                y,
                color=self.MASK_COLOR,
                batch=self._mask_batch,
            )
        )

    def _render_pointcloud(self, pointcloud_vertexes, pointcloud_colors):
        np.array(self._pointcloud_vlist.position, copy=False)[:] = (
            pointcloud_vertexes.ravel()
        )
        np.array(self._pointcloud_vlist.colors, copy=False)[:] = (
            pointcloud_colors.ravel()
        )

        # Look at the origin, sweep on the x-axis
        self.view = Mat4.look_at(
            position=Vec3(
                math.sin(self._flyby_theta) * 2,
                0,
                -2,
            ),
            target=Vec3(0, 0, 0),
            up=Vec3(0, -1, 0),
        )
        # Offset the view to the second half of the screen
        self.view = self.view.scale((0.7, 0.7, 1))
        self.view = self.view.translate((0, 1.6, 0))

        self._frustrum_batch.draw()
        self._pointcloud_batch.draw()


def run():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname).4s {%(name)s:%(lineno)s} %(message)s",
    )
    app = AppWindow()
    pyglet.app.run()
    app.stop()


if __name__ == "__main__":
    run()

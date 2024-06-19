# Artificial Senses

This software demonstrates how computer vision sees the environment 
and can detect humans using machine learning.

It uses an [Intel Realsense camera](https://www.intelrealsense.com/)
for visible light and depth acquisition (tested with a
[D455F](https://www.intelrealsense.com/depth-camera-d455f/)),
[YOLOv8](https://www.ultralytics.com/yolo) for segmentation and human 
detection (using a pre-trained [COCO dataset])(https://cocodataset.org/))
and [Pyglet](https://pyglet.org/) to visualize a flyby of a point cloud.

## Setup

Note: python 3.11 is required.

Using [poetry](https://python-poetry.org/):

```
$ poetry install
$ poetry run artificial-senses
```

Using pip:

```
$ python3 -m venv .venv && source .venv/bin/activate
$ pip install .
$ artificial-senses
```

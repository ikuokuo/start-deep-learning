#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
import argparse
import os

import cv2 as cv
import mmcv
from mmskeleton.apis import init_pose_estimator, inference_pose_estimator
from mmskeleton.processor import pose_demo


class Camera(object):

  def __init__(self, index=0, width=None, height=None, fps=None):
    cam = cv.VideoCapture(index)
    width is None or cam.set(cv.CAP_PROP_FRAME_WIDTH, width)
    height is None or cam.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    fps is None or cam.set(cv.CAP_PROP_FPS, fps)
    print(f"CAM: {cam.get(cv.CAP_PROP_FRAME_WIDTH)}x{cam.get(cv.CAP_PROP_FRAME_HEIGHT)} {cam.get(cv.CAP_PROP_FPS)}")
    self._cam = cam

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.release()

  def isOpened(self):
    return self._cam.isOpened()

  def read(self):
    return self._cam.read()

  def reads(self):
    while self.isOpened():
      success, frame = self.read()
      if success:
        yield frame
      else:
        break

  def release(self):
    if not self._cam:
      return
    self._cam.release()
    self._cam = None


def main():
  args = parse_args()

  win_name = args.win_name
  cv.namedWindow(win_name, cv.WINDOW_NORMAL)

  with Camera(args.cam_idx, args.cam_width, args.cam_height, args.cam_fps) as cam:
    cfg = mmcv.Config.fromfile(args.cfg_file)
    detection_cfg = cfg["detection_cfg"]

    print("Loading model ...")
    model = init_pose_estimator(**cfg, device=0)
    print("Loading model done")

    for frame in cam.reads():
      res = inference_pose_estimator(model, frame)

      res_image = pose_demo.render(
          frame, res["joint_preds"], res["person_bbox"],
          detection_cfg.bbox_thre)

      cv.imshow(win_name, res_image)

      key = cv.waitKey(1) & 0xFF
      if key == 27 or key == ord("q"):
        break

  cv.destroyAllWindows()


def parse_args():
  cur_dir = os.path.dirname(os.path.realpath(__file__))

  parser = argparse.ArgumentParser(usage="python scripts/coco2yolov5.py <options>")

  parser.add_argument("--win_name", type=str, default="webcam",
      help="window name, default: %(default)s")

  parser.add_argument("--cam_idx", type=int, default=0,
      help="camera index, default: %(default)s")
  parser.add_argument("--cam_width", type=int, default=None,
      help="camera width, default: %(default)s")
  parser.add_argument("--cam_height", type=int, default=None,
      help="camera height, default: %(default)s")
  parser.add_argument("--cam_fps", type=int, default=None,
      help="camera fps, default: %(default)s")

  parser.add_argument("--cfg_file", type=str,
      default=os.path.join(cur_dir, "configs/apis/pose_estimator.cascade_rcnn+hrnet.yaml"),
      help="config file, default: %(default)s")

  args = parser.parse_args()

  print("Args")
  print(f"  win_name: {args.win_name}")
  print(f"  cam_idx: {args.cam_idx}")
  print(f"  cam_width: {args.cam_width}")
  print(f"  cam_height: {args.cam_height}")
  print(f"  cam_fps: {args.cam_fps}")
  print(f"  cfg_file: {args.cfg_file}")

  return args


if __name__ == "__main__":
  main()

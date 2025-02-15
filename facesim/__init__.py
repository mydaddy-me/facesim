from dataclasses import dataclass

import cv2
import dlib
import logging
import numpy as np
from cv2.typing import MatLike
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

def crop_bounding_rectangle(image, rect, ow: int, oh: int):
    aspect_ratio = ow / oh
    x, y, w, h = rect
    
    if w / h > aspect_ratio:
        new_w = w
        new_h = int(w / aspect_ratio)
    else:
        new_h = h
        new_w = int(h * aspect_ratio)

    if new_w < ow or new_h < oh:
        # logging.warning(f"Bounding box too small: {w}x{h}")
        return None

    center_x, center_y = x + w // 2, y + h // 2
    x_new = max(0, center_x - new_w // 2)
    y_new = max(0, center_y - new_h // 2)

    x_new = min(x_new, image.shape[1] - new_w)
    y_new = min(y_new, image.shape[0] - new_h)

    res = image[y_new:y_new + new_h, x_new:x_new + new_w]
    return cv2.resize(res, (ow, oh))


detector = dlib.get_frontal_face_detector()  # type: ignore

def crop_face(image:MatLike):
    """ 
    Crop the face from the image. 
    """
    w, h = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    
    if not faces:
        return None
    
    face = faces[0]
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    return crop_bounding_rectangle(image, (x, y, w, h), 256, 256)


@dataclass
class parts:
    left_eyebrow: np.ndarray
    # right_eyebrow: np.ndarray
    # left_eye: np.ndarray
    # right_eye: np.ndarray
    nose: np.ndarray
    lips: np.ndarray


predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def crop_parts(face256x256):
    """
    Crop parts from the face image.
    input size is 256x256 
    """
    assert face256x256.shape[:2] == (256, 256), \
        f"Input image must be 256x256 got {face256x256.shape[:2]}"

    gray = cv2.cvtColor(face256x256, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    assert len(faces) == 1, f"Expected 1 face"
    landmarks = predictor(gray, faces[0])

    left_eyebrow_pts = [
        landmarks.part(n)
        for n in range(17, 22)]

    right_eyebrow_pts = [
        landmarks.part(n)
        for n in range(22, 27)]

    left_eye_pts = [
        landmarks.part(n)
        for n in range(36, 42)]

    right_eye_pts = [
        landmarks.part(n)
        for n in range(42, 48)]

    nose_pts = [
        landmarks.part(n)
        for n in range(27, 36)]

    lips_pts = [
        landmarks.part(n)
        for n in range(48, 68)]

    def extract_bb(pts, ow=48, oh=48):
        x_min = min(pt.x for pt in pts)
        y_min = min(pt.y for pt in pts)
        x_max = max(pt.x for pt in pts)
        y_max = max(pt.y for pt in pts)
        rect = (x_min, y_min, x_max - x_min, y_max - y_min)
        return crop_bounding_rectangle(face256x256, rect, ow, oh)

    return parts(
        left_eyebrow=extract_bb(left_eyebrow_pts, oh=16),
        # right_eyebrow=extract_bb(right_eyebrow_pts, oh=16),
        # left_eye=extract_bb(left_eye_pts, oh=24),
        # right_eye=extract_bb(right_eye_pts, oh=24),
        nose=extract_bb(nose_pts, ow=24),
        lips=extract_bb(lips_pts, oh=16))



import logging
from dataclasses import dataclass

import cv2
import dlib
from cv2.typing import MatLike
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

Rect = tuple[int, int, int, int]


def pad(rect: Rect, pad: float = 0.1):
    x, y, w, h = rect

    pad_x = int(w * pad)
    pad_y = int(h * pad)

    return x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y


def crop_resize(image: MatLike, rect: Rect, size: int = 64):
    try:
        x, y, w, h = rect
        d = max(w, h) // 2
        cx = x + d
        cy = y + d
        res = image[cy - d:cy + d, cx - d:cx + d]
        return cv2.resize(res, (size, size))
    except Exception:
        return None


detector = dlib.get_frontal_face_detector()  # type: ignore


def crop_face(image: MatLike):
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
    return crop_resize(image, (x, y, w, h), 256)


@dataclass
class parts:
    left_eyebrow: MatLike | None
    right_eyebrow: MatLike | None
    left_eye: MatLike | None
    right_eye: MatLike | None
    nose: MatLike | None
    lips: MatLike | None


predictor = dlib.shape_predictor(  # type: ignore
    "shape_predictor_68_face_landmarks.dat")


def crop_parts(face256x256):
    """
    Crop parts from the face image.
    input size is 256x256 
    """
    assert face256x256.shape[:2] == (256, 256), \
        f"Input image must be 256x256 got {face256x256.shape[:2]}"

    gray = cv2.cvtColor(face256x256, cv2.COLOR_BGR2GRAY)
    # faces = detector(gray)
    # assert len(faces) == 1, f"Expected 1 face, got {len(faces)}"
    landmarks = predictor(gray, dlib.rectangle(0, 0, 256, 256))  # type: ignore

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

    def extract_bb(pts):
        x_min = min(pt.x for pt in pts)
        y_min = min(pt.y for pt in pts)
        x_max = max(pt.x for pt in pts)
        y_max = max(pt.y for pt in pts)
        rect = (x_min, y_min, x_max - x_min, y_max - y_min)
        return crop_resize(face256x256, rect)

    return parts(
        left_eyebrow=extract_bb(left_eyebrow_pts),
        right_eyebrow=extract_bb(right_eyebrow_pts),
        left_eye=extract_bb(left_eye_pts),
        right_eye=extract_bb(right_eye_pts),
        nose=extract_bb(nose_pts),
        lips=extract_bb(lips_pts))

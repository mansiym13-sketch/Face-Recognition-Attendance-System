import os
import cv2
import numpy as np
from PIL import Image


def iter_image_paths(images_root: str):
    if not os.path.isdir(images_root):
        return []
    person_dirs = [os.path.join(images_root, d) for d in os.listdir(images_root) if os.path.isdir(os.path.join(images_root, d))]
    for d in person_dirs:
        for f in os.listdir(d):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                yield os.path.join(d, f)


def parse_enrollment_from_filename(path: str) -> int:
    # Filename pattern: {name}_{enrollment}_{n}.jpg
    base = os.path.basename(path)
    try:
        parts = os.path.splitext(base)[0].split("_")
        # enrollment is assumed to be the second part
        return int(parts[1])
    except Exception as e:
        raise ValueError(f"Cannot parse enrollment from {base}") from e


def train_model(haarcascade_path: str, images_root: str, model_path: str) -> int:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    ids = []

    paths = list(iter_image_paths(images_root))
    if not paths:
        raise RuntimeError("No training images found. Enroll first.")

    for p in paths:
        pilImage = Image.open(p).convert("L")
        image_np = np.array(pilImage, "uint8")
        Id = parse_enrollment_from_filename(p)
        faces.append(image_np)
        ids.append(Id)

    recognizer.train(faces, np.array(ids))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    recognizer.save(model_path)
    return len(paths)

import numpy as np
import cv2
from deepface import DeepFace

ARC = DeepFace.build_model("ArcFace")


def _decode_image(image_bytes: bytes, max_side: int = 1024) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image data")

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]

    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_LINEAR
        )

    return img


# For enrollment (single face)
def extract_embedding(image_bytes: bytes, strict: bool = False):
    img = _decode_image(image_bytes)

    reps = DeepFace.represent(
        img_path=img,
        model_name="ArcFace",
        detector_backend="yunet",
        enforce_detection=strict
    )

    return reps[0]["embedding"]


# For recognition (multiple faces)
def extract_embeddings(image_bytes: bytes):
    img = _decode_image(image_bytes)

    reps = DeepFace.represent(
        img_path=img,
        model_name="ArcFace",
        detector_backend="yunet",
        enforce_detection=False
    )

    results = []

    for e in reps:
        area = e["facial_area"]

        results.append({
            "embedding": e["embedding"],
            "bbox": {
                "x": int(area["x"]),
                "y": int(area["y"]),
                "w": int(area["w"]),
                "h": int(area["h"])
            }
        })

    return results
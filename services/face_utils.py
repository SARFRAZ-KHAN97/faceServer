import numpy as np
import cv2
from deepface import DeepFace

def extract_embedding(image_bytes):
    """
    For /enroll-student
    Expects one clear face, returns single embedding.
    Uses MTCNN for detection, works directly on numpy array.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image data")

    embeddings = DeepFace.represent(
        img_path=img,                # 👈 pass numpy array
        model_name="Facenet",
        detector_backend="mtcnn",
        enforce_detection=True
    )

    return embeddings[0]["embedding"]




def extract_embeddings(image_bytes):
    """
    For /recognize
    Detects multiple faces and returns list of {embedding, bbox}.
    Uses MTCNN for detection, no temp files.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image data")

    embeddings = DeepFace.represent(
        img_path=img,
        model_name="Facenet",
        detector_backend="mtcnn",
        enforce_detection=False
    )

    results = []
    for e in embeddings:
        facial_area = e["facial_area"]
        bbox = {
            "x": int(facial_area["x"]),
            "y": int(facial_area["y"]),
            "w": int(facial_area["w"]),
            "h": int(facial_area["h"]),
        }

        results.append({
            "embedding": e["embedding"],
            "bbox": bbox
        })

    return results


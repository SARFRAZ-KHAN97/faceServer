from fastapi import APIRouter, UploadFile, Form
import json
import os
from services.face_utils import extract_embedding

router = APIRouter()

EMBEDDINGS_FILE = "embeddings/student_embeddings.json"

# Ensure folder
os.makedirs("embeddings", exist_ok=True)

# Ensure file
if not os.path.exists(EMBEDDINGS_FILE) or os.path.getsize(EMBEDDINGS_FILE) == 0:
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump({}, f)

# Load once in memory
with open(EMBEDDINGS_FILE, "r") as f:
    KNOWN_EMBEDDINGS = json.load(f)


@router.post("/enroll-student")
async def enroll_student(
    studentId: str = Form(...),
    file: UploadFile = None
):
    if not file:
        return {
            "status": "failure",
            "message": "No image uploaded"
        }

    image_bytes = await file.read()

    try:
        embedding = extract_embedding(image_bytes, strict=True)

    except Exception as e:
        return {
            "status": "failure",
            "message": str(e)
        }

    # Update memory instantly
    KNOWN_EMBEDDINGS[studentId] = {
        "embedding": embedding
    }

    # Save file
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(KNOWN_EMBEDDINGS, f)

    return {
        "status": "success",
        "message": "Student enrolled successfully",
        "studentId": studentId
    }
from fastapi import APIRouter, UploadFile, Form
import json
import os
from services.face_utils import extract_embedding


router= APIRouter()

EMBEDDINGS_FILE= "embeddings/student_embeddings.json"


# Ensure directory and file exist
if not os.path.exists("embeddings"):
    os.makedirs("embeddings")

# If file doesn’t exist or is empty, initialize it with {}
if not os.path.exists(EMBEDDINGS_FILE) or os.path.getsize(EMBEDDINGS_FILE) == 0:
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump({}, f)


@router.post("/enroll-student")
async def enroll_student(studentId: str = Form(...), file: UploadFile = None):
    print("hii")
    if not file:
        return {"status": "failure", "message": "No image uploaded"}

    # read image
    image_bytes = await file.read()

    try:
        embedding = extract_embedding(image_bytes)
    except Exception as e:
        return {"status": "failure", "message": str(e)}

    # load existing embeddings
    with open(EMBEDDINGS_FILE, "r") as f:
        data = json.load(f)

    # update/add student
    data[studentId] = {"embedding": embedding}

    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(data, f)

    return {"status": "success", "message": "Student enrolled successfully", "studentId": studentId}

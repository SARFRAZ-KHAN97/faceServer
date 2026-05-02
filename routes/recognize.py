from fastapi import APIRouter, UploadFile, Form
from routes.enroll import KNOWN_EMBEDDINGS
from services.face_utils import extract_embeddings
from services.compare_utils import is_match

router = APIRouter()


@router.post("/recognize")
async def recognize_student(
    lectureId: str = Form(...),
    file: UploadFile = None
):
    if not file:
        return {
            "status": "failure",
            "message": "No image uploaded"
        }

    image_bytes = await file.read()

    try:
        faces = extract_embeddings(image_bytes)
    except Exception as e:
        return {
            "status": "failure",
            "message": str(e)
        }

    if not KNOWN_EMBEDDINGS:
        return {
            "status": "failure",
            "message": "No students enrolled yet"
        }

    present_students = []
    results = []

    for face in faces:
        embedding = face["embedding"]
        bbox = face["bbox"]

        match_id, score = is_match(embedding, KNOWN_EMBEDDINGS)

        if match_id:
            present_students.append(match_id)
            results.append({
                "studentId": match_id,
                "similarity": score,
                "bbox": bbox
            })
        else:
            results.append({
                "studentId": None,
                "similarity": None,
                "bbox": bbox,
                "message": "No match found"
            })

    return {
        "status": "success",
        "presentStudents": list(set(present_students)),
        "recognized": results
    }
from fastapi import APIRouter, UploadFile, Form
import json, requests
from services.face_utils import extract_embeddings
from services.compare_utils import is_match

router = APIRouter()
EMBEDDINGS_FILE = "embeddings/student_embeddings.json"

NODE_SERVER_URL = "http://localhost:4040/attd/mark-batch"  # Node endpoint

@router.post("/recognize")
async def recognize_student(
    lectureId: str = Form(...),
    file: UploadFile = None
):
    if not file:
        return {"status": "failure", "message": "No image uploaded"}

    image_bytes = await file.read()

    try:
        faces = extract_embeddings(image_bytes)  # embeddings + bboxes
    except Exception as e:
        return {"status": "failure", "message": str(e)}

    with open(EMBEDDINGS_FILE, "r") as f:
        known_embeddings = json.load(f)

    if not known_embeddings:
        return {"status": "failure", "message": "No students enrolled yet"}

    present_students = []
    results = []

    for face in faces:
        embedding = face["embedding"]
        bbox = face["bbox"]

        match_id, score = is_match(embedding, known_embeddings)
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

    # ---- send to Node server ----
    try:
        resp = requests.post(NODE_SERVER_URL, json={
            "lectureId": lectureId,
            "presentStudents": present_students
        })
        node_response = resp.json()
    except Exception as e:
        node_response = {"status": "failure", "message": f"Could not update Node server: {e}"}

    return {
        "status": "success",
        "recognized": results,
        "attendanceUpdate": node_response
    }

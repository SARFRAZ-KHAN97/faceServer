from fastapi import APIRouter, UploadFile, Form, BackgroundTasks
import httpx
from routes.enroll import KNOWN_EMBEDDINGS
from services.face_utils import extract_embeddings
from services.compare_utils import is_match

router = APIRouter()

NODE_SERVER_URL = "http://4.194.252.156:4040/attd/mark-batch"


async def post_attendance(lecture_id, present_students):
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                NODE_SERVER_URL,
                json={
                    "lectureId": lecture_id,
                    "presentStudents": present_students
                },
                timeout=5
            )
    except:
        pass


@router.post("/recognize")
async def recognize_student(
    background_tasks: BackgroundTasks,
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
        match_id, score = is_match(
            face["embedding"],
            KNOWN_EMBEDDINGS
        )

        if match_id:
            present_students.append(match_id)

            results.append({
                "studentId": match_id,
                "similarity": score,
                "bbox": face["bbox"]
            })

        else:
            results.append({
                "studentId": None,
                "similarity": None,
                "bbox": face["bbox"],
                "message": "No match found"
            })

    background_tasks.add_task(
        post_attendance,
        lectureId,
        present_students
    )

    return {
        "status": "success",
        "recognized": results,
        "attendanceUpdate": "Processing in background"
    }
from fastapi import FastAPI
from routes import enroll, recognize



app= FastAPI(title="Face Recognition server", version="1.0")



app.include_router(enroll.router, prefix="/api")
app.include_router(recognize.router, prefix="/api")



@app.get("/")
def root():
    return {"message": "Face Recognition API is running"}

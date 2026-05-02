from fastapi import FastAPI
from routes import enroll, recognize
from fastapi.middleware.cors import CORSMiddleware

app= FastAPI(title="Face Recognition server", version="1.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      
        "http://127.0.0.1:3000",      
        "http://4.194.252.156:3000"                        
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(enroll.router, prefix="/api")
app.include_router(recognize.router, prefix="/api")



@app.get("/")
def root():
    return {"message": "Face Recognition API is running"}

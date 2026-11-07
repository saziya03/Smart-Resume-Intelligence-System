
from fastapi import FastAPI

app = FastAPI(
    title="Smart Resume Intelligence System",
    description="AI-powered Resume Screening and Matching API",
    version="1.0.0"
)

@app.get("/")
def home():
    return {
        "message": "Welcome to Smart Resume Intelligence System API"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "project": "Smart Resume Intelligence System"
    }

from fastapi import FastAPI

from app.routers.estimations_route import router as router

app = FastAPI(
    title="Estimator CAG API",
    description="API para generar estimaciones de proyectos de software usando modelos LLM.",
    version="0.1.0",
)
app.include_router(prefix="/api/v1", router=router)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def main():
    return {"message": "Please check the API documentation at endpoint /docs."}

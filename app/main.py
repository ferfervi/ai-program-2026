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
    #logging for debugging
    print("Health check endpoint was called.")
    return {"status": "healthy"}

@app.get("/")
def main():
    # logging for debugging
    print("Root endpoint was called.")
    return {"message": "Please check the API documentation at endpoint /docs."}

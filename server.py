from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from predict import predict_condition

app = FastAPI(title="Condition Detection API")

class PredictionRequest(BaseModel):
    text: str
    ling_features: Optional[List[float]] = None
    threshold: Optional[float] = 0.45

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    label: str
    threshold_used: float
    flag: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Condition Detection API is running. Use the /predict endpoint."}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        result = predict_condition(
            text=request.text,
            ling_features=request.ling_features,
            threshold=request.threshold
        )

        if isinstance(result, str):
            raise HTTPException(status_code=500, detail=result)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the server
    uvicorn.run(app, host="127.0.0.1", port=8000)

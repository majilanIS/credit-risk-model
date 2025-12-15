from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import mlflow.pyfunc
import pandas as pd
from src.api.pydantic_models import TransactionData, PredictionResponse

app = FastAPI(title="Credit Risk API")

# Load model from MLflow registry (Production stage recommended)
model = mlflow.pyfunc.load_model("models:/CreditRiskModel/Production")

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionData):
    try:
        df = pd.DataFrame([transaction.model_dump()])
        risk_prob = model.predict(df)[0]
        return PredictionResponse(risk_probability=float(risk_prob))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}

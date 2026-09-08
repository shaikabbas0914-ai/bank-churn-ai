import os
import sys
import json
import io
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional

# Add project root and src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.predict import (
    predict_single,
    predict_batch,
    calculate_customer_clv,
    explain_customer_prediction,
    get_global_feature_importance,
    simulate_retention_scenario,
    generate_sample_batch_csv,
    generate_executive_report_text
)

app = FastAPI(
    title="Bank Customer Churn AI API",
    description="REST API for real-time customer churn intelligence, XAI, and simulation",
    version="2.0.0"
)

# Enable CORS for React frontend (Vite default is http://localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CustomerInput(BaseModel):
    CreditScore: int = 650
    Geography: str = "France"
    Gender: str = "Female"
    Age: int = 42
    Tenure: int = 4
    Balance: float = 85000.0
    NumOfProducts: int = 1
    HasCrCard: int = 1
    IsActiveMember: int = 0
    EstimatedSalary: float = 95000.0
    model_name: Optional[str] = "ANN"
    threshold: Optional[float] = 0.50

class WhatIfRequest(BaseModel):
    customer: Dict[str, Any]
    adjustments: Dict[str, Any]
    model_name: Optional[str] = "ANN"
    threshold: Optional[float] = 0.50

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "Bank Churn AI Backend"}

@app.post("/api/predict/single")
def api_predict_single(payload: CustomerInput):
    try:
        data_dict = payload.model_dump()
        model_name = data_dict.pop("model_name", "ANN")
        threshold = data_dict.pop("threshold", 0.50)

        # Run inference and analytics
        prediction_result = predict_single(data_dict, model_name=model_name, threshold=threshold)
        clv_result = calculate_customer_clv(data_dict)
        explanations = explain_customer_prediction(data_dict, model_name=model_name)
        report_text = generate_executive_report_text(data_dict, prediction_result, clv_result, explanations)

        return {
            "prediction": prediction_result,
            "clv": clv_result,
            "explanations": explanations,
            "report_text": report_text
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/predict/what-if")
def api_what_if_simulation(payload: WhatIfRequest):
    try:
        sim_result = simulate_retention_scenario(
            payload.customer,
            payload.adjustments,
            model_name=payload.model_name,
            threshold=payload.threshold
        )
        return sim_result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/predict/batch")
async def api_predict_batch(
    file: UploadFile = File(...),
    model_name: str = Form("ANN"),
    threshold: float = Form(0.50)
):
    try:
        contents = await file.read()
        df_input = pd.read_csv(io.BytesIO(contents))
        
        df_results = predict_batch(df_input, model_name=model_name, threshold=threshold)
        
        # Aggregate portfolio summary
        total_cust = len(df_results)
        high_risk_df = df_results[df_results["Risk Category"] == "High Risk"]
        high_risk_count = len(high_risk_df)
        churn_rate = (high_risk_count / total_cust) * 100 if total_cust > 0 else 0
        
        total_balance_at_risk = float(high_risk_df["Balance"].sum()) if "Balance" in high_risk_df.columns else 0.0
        total_rev_at_risk = float(df_results["Revenue At Risk ($)"].sum()) if "Revenue At Risk ($)" in df_results.columns else 0.0
        
        # Risk distribution counts
        risk_dist = df_results["Risk Category"].value_counts().to_dict()
        
        # Geo distribution of revenue at risk
        geo_rev = {}
        if "Geography" in df_results.columns and "Revenue At Risk ($)" in df_results.columns:
            geo_rev = df_results.groupby("Geography")["Revenue At Risk ($)"].sum().round(2).to_dict()

        # Convert to records JSON
        records = df_results.head(100).to_dict(orient="records")

        return {
            "summary": {
                "total_customers": total_cust,
                "high_risk_count": high_risk_count,
                "churn_rate_pct": round(churn_rate, 2),
                "total_balance_at_risk": round(total_balance_at_risk, 2),
                "total_revenue_at_risk": round(total_rev_at_risk, 2),
                "projected_savings_20pct": round(total_rev_at_risk * 0.20, 2),
                "risk_distribution": risk_dist,
                "geo_revenue_at_risk": geo_rev
            },
            "records": records,
            "total_records_count": len(df_results)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/data/sample-csv")
def api_get_sample_csv(n: int = 50):
    try:
        sample_df = generate_sample_batch_csv(n_samples=n)
        stream = io.StringIO()
        sample_df.to_csv(stream, index=False)
        response = Response(content=stream.getvalue(), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=sample_bank_churn_batch.csv"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/comparison")
def api_model_comparison():
    path = "models/model_comparison.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
            return data
    raise HTTPException(status_code=404, detail="Model comparison data not found.")

@app.get("/api/models/feature-importance")
def api_feature_importance():
    try:
        feat_dict = get_global_feature_importance()
        res = {}
        for k, v in feat_dict.items():
            res[k] = v.to_dict(orient="records")
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/eda/demographics")
def api_eda_demographics():
    path = "data/Churn_Modelling.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        sample_3d = df.sample(n=min(1200, len(df)), random_state=42)[
            ["CreditScore", "Age", "Balance", "Geography", "Gender", "Exited"]
        ].to_dict(orient="records")
        
        geo_churn = df.groupby("Geography")["Exited"].mean().mul(100).round(2).to_dict()
        prod_churn = df.groupby("NumOfProducts")["Exited"].mean().mul(100).round(2).to_dict()
        gender_churn = df.groupby("Gender")["Exited"].mean().mul(100).round(2).to_dict()

        return {
            "sample_3d": sample_3d,
            "geo_churn": geo_churn,
            "prod_churn": prod_churn,
            "gender_churn": gender_churn
        }
    raise HTTPException(status_code=404, detail="Dataset not found.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

# === API FASTAPI SECURISÉE POUR LE SCORING CLIENT AVEC SHAP ===
# Objectif : recevoir un client (JSON), retourner un score de défaut ET une explication SHAP

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Dict
import pandas as pd
import joblib
import shap
import os
import numpy as np

# Charger le modèle pipeline
pipeline_model = joblib.load("model.pkl")
scaler = pipeline_model.named_steps['scaler']
clf = pipeline_model.named_steps['clf']
model_features = scaler.get_feature_names_out()

# Clé d'API sécurisée
API_KEY = os.getenv("API_KEY")

# FastAPI app
app = FastAPI(title="API IA Crédit Néo-Banque", version="1.1")

# Modèle d'entrée
class ClientData(BaseModel):
    features: Dict[str, float]

# Auth middleware
@app.middleware("http")
async def verify_token(request: Request, call_next):
    if request.url.path != "/":
        auth = request.headers.get("Authorization")
        if not auth or not auth.startswith("Bearer ") or auth.split("Bearer ")[1] != API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing API token")
    return await call_next(request)

@app.get("/")
def read_root():
    return {"message": "API de scoring + SHAP disponible."}

@app.post("/predict")
def predict_score(data: ClientData):
    input_df = pd.DataFrame([data.features])
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_features]
    input_scaled = scaler.transform(input_df)
    score = clf.predict_proba(input_scaled)[0][1]
    decision = "Éligible" if score < 0.4 else "Risque modéré" if score < 0.7 else "Risque élevé"
    return {"score": round(float(score), 4), "decision": decision}

@app.post("/explain")
def explain_prediction(data: ClientData):
    try:
        # Création du DataFrame à partir des données reçues
        input_df = pd.DataFrame([data.features])

        # Compléter avec les colonnes manquantes (très important)
        for col in model_features:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reordonner comme à l'entraînement
        input_df = input_df[model_features]

        # Log pour debug
        print("✅ Colonnes reçues :", input_df.columns.tolist())
        print("✅ Shape input SHAP :", input_df.shape)

        # Transformation
        input_scaled = scaler.transform(input_df)

        # Initialiser SHAP correctement
        explainer = shap.Explainer(clf)
        shap_values = explainer(input_scaled)

        # Extraire les contributions du premier client
        shap_local_values = shap_values[0].values
        local_shap = dict(zip(model_features, shap_local_values))

        # Sélectionner les plus impactantes
        top_features = sorted(local_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

        return {
            "top_contributions": [
        {"feature": str(f), "impact": round(float(v), 4)} for f, v in top_features
]}

    except Exception as e:
        print("❌ Erreur SHAP :", str(e))
        raise HTTPException(status_code=500, detail=f"Erreur SHAP : {str(e)}")

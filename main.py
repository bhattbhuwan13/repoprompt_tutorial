from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os

app = FastAPI()

# Define the input schema (all columns except "quality" and "Id")
class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict_wine_quality(wine: WineInput):
    # Load the trained model
    model_path = os.path.join(os.path.dirname(__file__), "models", "decision_tree_wineqt.joblib")
    clf = joblib.load(model_path)

    # Prepare input for prediction
    features = np.array([[wine.fixed_acidity, wine.volatile_acidity, wine.citric_acid, wine.residual_sugar,
                          wine.chlorides, wine.free_sulfur_dioxide, wine.total_sulfur_dioxide, wine.density,
                          wine.pH, wine.sulphates, wine.alcohol]])
    prediction = clf.predict(features)
    return {"predicted_quality": int(prediction[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

with open('water_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = FastAPI()

class WaterInput(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

@app.post("/predict")
def predict(data: WaterInput):
    input_data = np.array([[
        data.ph, data.Hardness, data.Solids, data.Chloramines,
        data.Sulfate, data.Conductivity, data.Organic_carbon,
        data.Trihalomethanes, data.Turbidity
    ]])
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    return {"Potable": bool(pred)}

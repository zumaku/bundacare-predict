from fastapi import FastAPI
from pydantic import BaseModel
import random

app = FastAPI()

# Model request body
class PredictRequest(BaseModel):
    image_url: str

@app.get("/")
def home():
    return {"message": "BundaCare API - Siap melakukan prediksi"}

@app.post("/predict")
def predict(request: PredictRequest):
    protein = round(random.uniform(0, 100), 2)      # gram
    karbohidrat = round(random.uniform(0, 200), 2)  # gram
    lemak = round(random.uniform(0, 100), 2)        # gram

    kalori = round((protein * 4) + (karbohidrat * 4) + (lemak * 9), 2)

    return {
        "food_name": "Makanan Baru",
        "image_url": request.image_url,
        "protein": protein,
        "carbohydrate": karbohidrat,
        "fat": lemak,
        "calories": kalori
    }

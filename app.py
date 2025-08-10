from fastapi import FastAPI
import random

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Siap melakukan prediksi"}

@app.post("/predict")
def predict():
    protein = round(random.uniform(0, 100), 2)      # gram
    karbohidrat = round(random.uniform(0, 200), 2)  # gram
    lemak = round(random.uniform(0, 100), 2)        # gram

    kalori = round((protein * 4) + (karbohidrat * 4) + (lemak * 9), 2)

    return {
        "protein": protein,
        "karbohidrat": karbohidrat,
        "lemak": lemak,
        "kalori": kalori
    }

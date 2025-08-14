from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from collections import defaultdict

app = FastAPI()

# =====================
# 1. Load Dataset Nutrisi
# =====================
nutrition_df = pd.read_csv("model/nutrition.csv")

# =====================
# 2. Load Model YOLO
# =====================
model = YOLO("model/best.pt")

# =====================
# 3. Model Request Body
# =====================
class PredictRequest(BaseModel):
    image_url: str

# =====================
# 4. Fungsi Hitung Nutrisi
# =====================
def calculate_nutrition(class_name, count):
    row = nutrition_df[nutrition_df["class"] == class_name].iloc[0]
    return {
        "kcal": row["kcal_per_100g"] * count,
        "protein": row["protein_g_per_100g"] * count,
        "fat": row["fat_g_per_100g"] * count,
        "carb": row["carb_g_per_100g"] * count
    }

# =====================
# 5. Endpoint Home
# =====================
@app.get("/")
def home():
    return {"message": "BundaCare API - Siap melakukan prediksi"}

# =====================
# 6. Endpoint Predict
# =====================
@app.post("/predict")
def predict(request: PredictRequest):
    # Download image
    response = requests.get(request.image_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    
    # Deteksi objek
    results = model(img)
    
    # Hitung jumlah tiap class
    counts = defaultdict(int)
    for box in results[0].boxes:
        class_id = int(box.cls)
        class_name = results[0].names[class_id]
        counts[class_name] += 1

    foods_list = []
    total_nutrition = {"protein": 0, "carbohydrate": 0, "fat": 0, "calories": 0}

    for food_name, count in counts.items():
        # Cari nutrisi di dataset
        row = nutrition_df[nutrition_df["class"].str.lower() == food_name.lower()]
        if not row.empty:
            protein = float(row["protein_g_per_100g"].values[0]) * count
            carbohydrate = float(row["carb_g_per_100g"].values[0]) * count
            fat = float(row["fat_g_per_100g"].values[0]) * count
            calories = float(row["kcal_per_100g"].values[0]) * count
        else:
            protein = carbohydrate = fat = calories = 0

        foods_list.append({
            "name": food_name,
            "count": count,
            "protein": protein,
            "carbohydrate": carbohydrate,
            "fat": fat,
            "calories": calories
        })

        total_nutrition["protein"] += protein
        total_nutrition["carbohydrate"] += carbohydrate
        total_nutrition["fat"] += fat
        total_nutrition["calories"] += calories

    return {
        "image_url": request.image_url,
        "foods": foods_list,
        "total": total_nutrition
    }
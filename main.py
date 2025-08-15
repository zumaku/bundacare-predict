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
# 4. Endpoint Home
# =====================
@app.get("/")
def home():
    return {"message": "BundaCare API - Siap melakukan prediksi"}

# =====================
# 5. Endpoint Predict (Dengan Bounding Box [x, y, w, h])
# =====================
@app.post("/predict")
def predict(request: PredictRequest):
    # Download image
    response = requests.get(request.image_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    
    # Deteksi objek
    results = model.predict(source=img, conf=0.25)
    
    # Siapkan struktur data untuk menyimpan hasil
    detected_foods = defaultdict(lambda: {'count': 0, 'bounding_boxes': []})

    # Iterasi melalui setiap kotak deteksi
    for box in results[0].boxes:
        # Ambil nama class
        class_id = int(box.cls)
        class_name = results[0].names[class_id]
        
        # Ambil koordinat [x1, y1, x2, y2]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        # Konversi ke format [x, y, w, h]
        x = x1
        y = y1
        w = x2 - x1  # Lebar (width)
        h = y2 - y1  # Tinggi (height)
        
        coordinates_xywh = [x, y, w, h]
        
        # Tambahkan data ke dictionary
        detected_foods[class_name]['count'] += 1
        detected_foods[class_name]['bounding_boxes'].append(coordinates_xywh)

    # Proses hasil untuk response API
    foods_list = []
    total_nutrition = {"protein": 0, "carbohydrate": 0, "fat": 0, "calories": 0}

    for food_name, data in detected_foods.items():
        count = data['count']
        
        # Cari nutrisi
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
            "bounding_boxes": data['bounding_boxes'],
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
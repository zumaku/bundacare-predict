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
# 5. Endpoint Predict (Dengan Debugging)
# =====================
@app.post("/predict")
def predict(request: PredictRequest):
    try:
        # Download image
        response = requests.get(request.image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        
        print(f"Image loaded: {img.size}")  # Debug: ukuran image
        
        # Deteksi objek
        results = model.predict(source=img, conf=0.25, verbose=True)
        
        # Debug: informasi hasil deteksi
        print(f"Number of detections: {len(results[0].boxes) if results[0].boxes else 0}")
        
        # Siapkan struktur data untuk menyimpan hasil
        detected_foods = defaultdict(lambda: {'count': 0, 'bounding_boxes': []})

        # Iterasi melalui setiap kotak deteksi
        if results[0].boxes is not None:
            for i, box in enumerate(results[0].boxes):
                # Ambil nama class
                class_id = int(box.cls)
                class_name = results[0].names[class_id]
                confidence = float(box.conf)
                
                # Ambil koordinat [x_min, y_min, x_max, y_max]
                coordinates = box.xyxy[0].cpu().numpy().tolist()
                
                # Debug: print setiap detection
                print(f"Detection {i+1}: {class_name} (conf: {confidence:.2f})")
                print(f"  Coordinates: {coordinates}")
                print(f"  Image bounds: width={img.width}, height={img.height}")
                
                # Validasi koordinat dalam bounds image
                x_min, y_min, x_max, y_max = coordinates
                if (x_min >= 0 and y_min >= 0 and 
                    x_max <= img.width and y_max <= img.height and
                    x_max > x_min and y_max > y_min):
                    
                    # Tambahkan data ke dictionary
                    detected_foods[class_name]['count'] += 1
                    detected_foods[class_name]['bounding_boxes'].append(coordinates)
                    print(f"  ✓ Valid bounding box added")
                else:
                    print(f"  ✗ Invalid bounding box, skipping")

        # Proses hasil untuk response API
        foods_list = []
        total_nutrition = {"protein": 0, "carbohydrate": 0, "fat": 0, "calories": 0}

        for food_name, data in detected_foods.items():
            count = data['count']
            
            # Cari nutrisi
            row = nutrition_df[nutrition_df["class"].str.lower() == food_name.lower()]
            if not row.empty:
                # Hitung nutrisi per 100g dikali jumlah item
                protein = float(row["protein_g_per_100g"].values[0]) * count
                carbohydrate = float(row["carb_g_per_100g"].values[0]) * count
                fat = float(row["fat_g_per_100g"].values[0]) * count
                calories = float(row["kcal_per_100g"].values[0]) * count
            else:
                protein = carbohydrate = fat = calories = 0
                print(f"Warning: No nutrition data found for {food_name}")

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

        print(f"Final response: {len(foods_list)} food types detected")
        
        return {
            "image_url": request.image_url,
            "image_dimensions": {"width": img.width, "height": img.height},  # Tambahan info
            "foods": foods_list,
            "total": total_nutrition
        }
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return {"error": str(e), "foods": [], "total": {"protein": 0, "carbohydrate": 0, "fat": 0, "calories": 0}}
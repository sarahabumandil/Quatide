from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import folium
import os
from .model_engine import QuatideEngine

app = FastAPI(title="Quatide Quantum API")
engine = QuatideEngine()

# تأمين مجلد للملفات الثابتة (الخريطة)
os.makedirs("app/static", exist_ok=True)

@app.get("/")
def home():
    return {"status": "Quatide System Online", "engine": "Hybrid Quantum-LSTM"}

@app.post("/predict/map")
async def generate_live_map(lat: float, lon: float, current_speed: float, wind: float, temp: float):
    """
    يستقبل بيانات اللحظة الحالية ويرسل خريطة تفاعلية
    """
    try:
        # محاكاة لبيانات الـ LSTM (7 أيام)
        # في التطبيق الحقيقي، هنا نربط مع API Copernicus
        dummy_lstm = np.random.rand(1, 7, 3) 
        raw_feats = np.array([[current_speed, wind, temp]])
        
        # حساب التوقع
        prediction = engine.predict_waste(dummy_lstm, raw_feats)
        
        # إنشاء الخريطة
        m = folium.Map(location=[lat, lon], zoom_start=10)
        color = 'red' if prediction > 80 else 'green'
        folium.CircleMarker(
            location=[lat, lon],
            radius=prediction/5,
            popup=f"Waste Level: {prediction:.2f} kg/km2",
            color=color, fill=True
        ).add_to(m)
        
        map_path = "app/static/live_map.html"
        m.save(map_path)
        
        return {"prediction": prediction, "map_url": "/static/live_map.html"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/static", StaticFiles(directory="app/static"), name="static")


# check_model.py
import os
import joblib
from pathlib import Path

MODEL_PATH = Path("models/weighted_model.pkl")

def check_model_integrity():
    if not MODEL_PATH.exists():
        print("❌ Model dosyası bulunamadı!")
        return False
    
    file_size = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"📁 Model boyutu: {file_size:.2f} MB")
    
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model başarıyla yüklendi!")
        print(f"📊 Model tipi: {type(model)}")
        return True
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        return False

if __name__ == "__main__":
    check_model_integrity()

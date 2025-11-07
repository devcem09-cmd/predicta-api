# ============================================================
# 🔹 split_model.py
# Büyük model dosyasını 20 MB'lık parçalara böler
# ============================================================

import os

INPUT_FILE = "weighted_model.pkl"  # bölünecek dosya
CHUNK_SIZE = 20 * 1024 * 1024      # 20 MB
OUTPUT_PREFIX = "weighted_model_part_"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"❌ Dosya bulunamadı: {INPUT_FILE}")

file_size = os.path.getsize(INPUT_FILE)
num_parts = (file_size // CHUNK_SIZE) + (1 if file_size % CHUNK_SIZE != 0 else 0)

print(f"📦 Kaynak dosya: {INPUT_FILE}")
print(f"📏 Boyut: {file_size / (1024*1024):.2f} MB")
print(f"🔹 Parça sayısı: {num_parts}")
print("=" * 50)

with open(INPUT_FILE, "rb") as f:
    part_num = 0
    while chunk := f.read(CHUNK_SIZE):
        part_name = f"{OUTPUT_PREFIX}{part_num:02d}"
        with open(part_name, "wb") as p:
            p.write(chunk)
        print(f"✅ {part_name} oluşturuldu ({len(chunk)/(1024*1024):.2f} MB)")
        part_num += 1

print("=" * 50)
print(f"🎉 Bölme işlemi tamamlandı. Toplam {part_num} parça üretildi.")

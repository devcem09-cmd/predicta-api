#!/usr/bin/env python3
"""
SSL sorunlarını atlayan Google Drive indirme testi
Kurumsal ağlarda SSL proxy sorunları için
"""

import urllib.request
import urllib.error
import ssl
import os

FILE_ID = "1nrza381tsPprpML7mAafd6DNU93y5BT7"
OUTPUT_PATH = "test_model.pkl"

def download_with_ssl_bypass(file_id, destination):
    """SSL doğrulamasını atlayarak indir (sadece test için!)"""
    
    # SSL doğrulamasını devre dışı bırak (SADECE TEST İÇİN!)
    ssl_context = ssl._create_unverified_context()
    
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    print(f"🔗 URL: {url}")
    print(f"📥 İndiriliyor (SSL doğrulama: KAPALI)...")
    print()
    
    try:
        # İlk istek
        request = urllib.request.Request(url)
        request.add_header('User-Agent', 'Mozilla/5.0')
        
        with urllib.request.urlopen(request, context=ssl_context) as response:
            # Büyük dosyalar için onay sayfasını kontrol et
            content = response.read()
            
            # Eğer HTML sayfası dönerse (onay gerekiyor), token'ı bul
            if b'<!DOCTYPE html>' in content[:100]:
                print("⚠️  Onay sayfası tespit edildi, token aranıyor...")
                
                # Token'ı HTML'den çıkar
                import re
                html_text = content.decode('utf-8', errors='ignore')
                
                # Confirm parametresini bul
                confirm_match = re.search(r'confirm=([^&"]+)', html_text)
                if confirm_match:
                    token = confirm_match.group(1)
                    print(f"✅ Token bulundu: {token[:20]}...")
                    
                    # Token ile tekrar dene
                    confirm_url = f"{url}&confirm={token}"
                    request = urllib.request.Request(confirm_url)
                    request.add_header('User-Agent', 'Mozilla/5.0')
                    
                    with urllib.request.urlopen(request, context=ssl_context) as confirmed_response:
                        content = confirmed_response.read()
                else:
                    print("❌ Token bulunamadı, direkt içeriği kaydet deneniyor...")
            
            # Dosyayı kaydet
            with open(destination, 'wb') as f:
                f.write(content)
            
            return len(content)
    
    except urllib.error.HTTPError as e:
        print(f"❌ HTTP Hatası: {e.code} - {e.reason}")
        raise
    except urllib.error.URLError as e:
        print(f"❌ URL Hatası: {e.reason}")
        raise

if __name__ == "__main__":
    try:
        print("=" * 60)
        print("🧪 SSL Bypass Google Drive İndirme Testi")
        print("=" * 60)
        print(f"📁 Dosya ID: {FILE_ID}")
        print(f"💾 Hedef: {OUTPUT_PATH}")
        print()
        print("⚠️  DİKKAT: SSL doğrulama KAPALI (sadece test için!)")
        print()
        
        # İndir
        size = download_with_ssl_bypass(FILE_ID, OUTPUT_PATH)
        
        # Doğrula
        if os.path.exists(OUTPUT_PATH):
            actual_size = os.path.getsize(OUTPUT_PATH)
            size_mb = actual_size / (1024 * 1024)
            
            print()
            print("=" * 60)
            print("📊 Sonuçlar:")
            print("=" * 60)
            print(f"✅ Dosya indirildi")
            print(f"📦 Boyut: {size_mb:.2f} MB ({actual_size:,} bytes)")
            
            if actual_size < 10000:  # 10KB'den küçükse
                print()
                print("⚠️  UYARI: Dosya çok küçük!")
                print()
                print("İçerik önizlemesi:")
                print("-" * 60)
                with open(OUTPUT_PATH, 'rb') as f:
                    content = f.read(1000)
                    try:
                        print(content.decode('utf-8', errors='ignore'))
                    except:
                        print(content[:500])
                print("-" * 60)
                print()
                
                # HTML içeriği var mı kontrol et
                with open(OUTPUT_PATH, 'rb') as f:
                    first_bytes = f.read(100)
                    if b'<!DOCTYPE' in first_bytes or b'<html' in first_bytes:
                        print("❌ HTML sayfası indirilmiş (dosya değil)")
                        print()
                        print("💡 Olası sebepler:")
                        print("1. Dosya paylaşımı KAPALI")
                        print("2. Dosya silinmiş veya taşınmış")
                        print("3. Google Drive erişim sorunu")
                        print()
                        print("🔧 Çözüm:")
                        print("1. Dosyayı tarayıcıdan açmayı dene:")
                        print(f"   https://drive.google.com/file/d/{FILE_ID}/view")
                        print("2. Paylaşım ayarlarını kontrol et")
                        print("3. Dosya ID'sinin doğru olduğunu kontrol et")
                    else:
                        print("⚠️  İçerik HTML değil ama çok küçük")
                
            else:
                print()
                print("✅ Dosya geçerli görünüyor!")
                print()
                print("🎉 Test BAŞARILI!")
                print()
                print("📝 Sonraki Adımlar:")
                print("1. Render.com'da bu SSL sorunu olmayacak")
                print("2. render.yaml'ı GitHub'a push et")
                print("3. Render otomatik build edecek")
            
            # Temizlik
            print()
            print("🗑️  Test dosyası siliniyor...")
            os.remove(OUTPUT_PATH)
            print("✅ Temizlendi")
            
        else:
            print("❌ Dosya bulunamadı")
            
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        
        print()
        print("💡 Alternatif Çözümler:")
        print()
        print("1. Tarayıcıdan manuel indir:")
        print(f"   https://drive.google.com/file/d/{FILE_ID}/view")
        print()
        print("2. Kurumsal ağdaysan, proxy ayarları gerekebilir")
        print()
        print("3. VPN kullanıyorsan, kapat ve tekrar dene")
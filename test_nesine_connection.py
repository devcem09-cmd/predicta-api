"""
Nesine API Debug Test Script
Bu script Nesine API'sini detaylı test eder ve sorunları gösterir
"""

import requests
import json
from datetime import datetime

def test_nesine_direct():
    """Direkt Nesine API'yi test et"""
    print("=" * 80)
    print("🧪 NESINE API DIRECT TEST")
    print("=" * 80)
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://www.nesine.com/",
            "Origin": "https://www.nesine.com",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "tr-TR,tr;q=0.9",
        }
        
        url = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"
        
        print(f"📡 URL: {url}")
        print(f"⏱️  Request gönderiliyor...")
        
        response = requests.get(url, headers=headers, timeout=15)
        
        print(f"✅ Status Code: {response.status_code}")
        print(f"📊 Response Size: {len(response.content)} bytes")
        
        if response.status_code == 200:
            data = response.json()
            
            # Veri yapısını incele
            print(f"\n📋 Response Keys: {list(data.keys())}")
            
            if "sg" in data:
                print(f"✅ 'sg' key bulundu")
                sg_keys = list(data["sg"].keys())
                print(f"📋 SG Keys: {sg_keys}")
                
                # EA ve CA kategorilerini kontrol et
                for category in ["EA", "CA"]:
                    if category in data["sg"]:
                        matches = data["sg"][category]
                        print(f"\n🎯 {category} Kategorisi:")
                        print(f"   Total Items: {len(matches)}")
                        
                        # Football maçlarını say
                        football_matches = [m for m in matches if m.get("GT") == 1]
                        print(f"   ⚽ Football Matches: {len(football_matches)}")
                        
                        if football_matches:
                            # İlk maçı detaylı göster
                            first = football_matches[0]
                            print(f"\n   📌 First Match Sample:")
                            print(f"      Home: {first.get('HN', 'N/A')}")
                            print(f"      Away: {first.get('AN', 'N/A')}")
                            print(f"      Date: {first.get('D', 'N/A')}")
                            print(f"      Time: {first.get('T', 'N/A')}")
                            print(f"      League: {first.get('LN', 'N/A')}")
                            
                            # Odds'u kontrol et
                            ma_data = first.get("MA", [])
                            print(f"      MA (Markets) Count: {len(ma_data)}")
                            
                            if ma_data:
                                for i, market in enumerate(ma_data[:3]):  # İlk 3 market
                                    mtid = market.get("MTID", "?")
                                    oca = market.get("OCA", [])
                                    print(f"      Market {i+1}: MTID={mtid}, OCA Count={len(oca)}")
                                    
                                    if mtid == 1:  # 1X2
                                        if len(oca) >= 3:
                                            print(f"         1X2 Odds:")
                                            print(f"         1: {oca[0].get('O', 'N/A')}")
                                            print(f"         X: {oca[1].get('O', 'N/A')}")
                                            print(f"         2: {oca[2].get('O', 'N/A')}")
                            
                            # İlk 3 maçı listele
                            print(f"\n   📋 First 3 Football Matches:")
                            for idx, m in enumerate(football_matches[:3], 1):
                                print(f"      {idx}. {m.get('HN', '?')} vs {m.get('AN', '?')}")
                                print(f"         League: {m.get('LN', '?')}")
                                print(f"         Date: {m.get('D', '?')} {m.get('T', '?')}")
                        else:
                            print(f"   ⚠️  No football matches found!")
                    else:
                        print(f"⚠️  {category} kategorisi bulunamadı!")
            else:
                print("❌ 'sg' key bulunamadı!")
                
            # Raw JSON'u dosyaya kaydet
            with open("nesine_raw_response.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Full response saved to: nesine_raw_response.json")
            
            return True
        else:
            print(f"❌ Bad status code: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT: Nesine API yanıt vermiyor (15 saniye)")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ CONNECTION ERROR: {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fetcher_module():
    """nesine_fetcher_fixed.py modülünü test et"""
    print("\n" + "=" * 80)
    print("🧪 NESINE FETCHER MODULE TEST")
    print("=" * 80)
    
    try:
        from nesine_fetcher_fixed import fetch_upcoming_matches
        print("✅ nesine_fetcher_fixed module imported successfully")
        
        print("📡 Fetching matches...")
        matches = fetch_upcoming_matches(force_refresh=True)
        
        print(f"📊 Returned {len(matches)} matches")
        
        if matches:
            print(f"\n✅ SUCCESS! Got {len(matches)} matches")
            
            # İlk 3 maçı göster
            print("\n📋 First 3 matches:")
            for idx, m in enumerate(matches[:3], 1):
                print(f"\n{idx}. {m['home_team']} vs {m['away_team']}")
                print(f"   League: {m.get('league_name', 'N/A')}")
                print(f"   Date: {m.get('date', 'N/A')}")
                print(f"   1X2 Odds: {m.get('odds_1x2', {})}")
                print(f"   O/U 2.5: {m.get('odds_over_under', {})}")
                print(f"   BTTS: {m.get('odds_btts', {})}")
            
            # İstatistikler
            valid_odds = sum(1 for m in matches if m.get('odds_1x2', {}).get('1', 0) > 0)
            print(f"\n📊 Statistics:")
            print(f"   Total matches: {len(matches)}")
            print(f"   With valid 1X2 odds: {valid_odds}")
            print(f"   Live matches: {sum(1 for m in matches if m.get('is_live', False))}")
            
            return True
        else:
            print("⚠️  No matches returned!")
            return False
            
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure nesine_fetcher_fixed.py is in the same directory")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_endpoint():
    """Local API endpoint'i test et"""
    print("\n" + "=" * 80)
    print("🧪 LOCAL API ENDPOINT TEST")
    print("=" * 80)
    
    try:
        api_url = "http://localhost:8000/api/matches/upcoming?force_refresh=true"
        print(f"📡 Testing: {api_url}")
        
        response = requests.get(api_url, timeout=10)
        print(f"✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"📊 Response Keys: {list(data.keys())}")
            print(f"✅ Success: {data.get('success', False)}")
            print(f"📋 Source: {data.get('source', 'unknown')}")
            print(f"🎯 Match Count: {data.get('count', 0)}")
            print(f"💾 Cached: {data.get('cached', False)}")
            
            matches = data.get('matches', [])
            if matches:
                print(f"\n📋 First match:")
                first = matches[0]
                print(f"   {first.get('home_team')} vs {first.get('away_team')}")
                print(f"   League: {first.get('league_name')}")
                print(f"   Odds: {first.get('odds')}")
            
            return True
        else:
            print(f"❌ Bad status: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to localhost:8000")
        print("💡 Make sure API is running: python main_weighted.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Tüm testleri çalıştır"""
    print("\n" + "🔬" * 40)
    print("NESINE API DEBUG TEST SUITE")
    print("🔬" * 40)
    print(f"⏰ Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = {}
    
    # Test 1: Direct API
    results['direct_api'] = test_nesine_direct()
    
    # Test 2: Fetcher Module
    results['fetcher_module'] = test_fetcher_module()
    
    # Test 3: Local API Endpoint
    results['api_endpoint'] = test_api_endpoint()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.upper()}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    print(f"\n🎯 Overall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("✅ All tests passed! Nesine integration is working.")
    else:
        print("\n⚠️  TROUBLESHOOTING TIPS:")
        
        if not results['direct_api']:
            print("  • Nesine API'ye erişim sorunu var")
            print("  • İnternet bağlantınızı kontrol edin")
            print("  • Firewall/proxy ayarlarını kontrol edin")
            print("  • Browser'da şu URL'yi test edin: https://www.nesine.com")
        
        if not results['fetcher_module']:
            print("  • nesine_fetcher_fixed.py dosyası eksik veya hatalı")
            print("  • Dosyanın doğru dizinde olduğundan emin olun")
            print("  • Import hatalarını kontrol edin")
        
        if not results['api_endpoint']:
            print("  • Local API çalışmıyor")
            print("  • API'yi başlatın: python main_weighted.py")
            print("  • Port 8000'in kullanılabilir olduğundan emin olun")
    
    print("\n" + "🔬" * 40)


if __name__ == "__main__":
    main()
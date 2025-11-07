import requests
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Cache for fetched matches
_matches_cache = None
_cache_timestamp = None
CACHE_DURATION = 300  # 5 minutes cache

def fetch_upcoming_matches(force_refresh: bool = False) -> List[Dict[str, Any]]:
    """
    Fetch upcoming matches from Nesine API with caching.
    
    Args:
        force_refresh: If True, bypass cache and fetch fresh data
        
    Returns:
        List of match dictionaries with proper odds and dates
    """
    global _matches_cache, _cache_timestamp
    
    # Check if we have valid cached data
    import time
    current_time = time.time()
    
    if (not force_refresh and 
        _matches_cache is not None and 
        _cache_timestamp is not None and
        (current_time - _cache_timestamp) < CACHE_DURATION):
        logger.info("✅ Using cached matches data")
        return _matches_cache
    
    # Fetch fresh data
    try:
        logger.info("🌐 Fetching upcoming matches from Nesine API...")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.nesine.com/",
            "Origin": "https://www.nesine.com",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        
        url = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        matches = []
        
        # Process all match categories
        for category in ["EA", "CA"]:  # EA: Prematch, CA: Live
            category_matches = data.get("sg", {}).get(category, [])
            
            for m in category_matches:
                if m.get("GT") != 1:  # Only football (GT=1)
                    continue
                
                # Extract and format date/time
                date_str = m.get("D", "")  # Format: "DD.MM.YYYY"
                time_str = m.get("T", "")  # Format: "HH:MM"
                match_date = _format_match_datetime(date_str, time_str)
                
                # Extract odds properly
                odds_data = _extract_all_odds(m.get("MA", []))
                
                # Extract league name
                league_name = m.get("LN", "") or m.get("LC", "") or "Unknown League"
                
                match_info = {
                    "home_team": m.get("HN", "Unknown Home"),
                    "away_team": m.get("AN", "Unknown Away"),
                    "league_name": league_name,
                    "league_code": m.get("LC", ""),
                    "league_id": m.get("LID", ""),
                    "date": match_date,
                    "time": time_str,
                    "match_id": m.get("C", ""),
                    "event_id": m.get("EV", ""),
                    "is_live": category == "CA",
                    
                    # Odds data
                    "odds_1x2": odds_data.get("1x2", {}),
                    "odds_over_under": odds_data.get("over_under", {}),
                    "odds_btts": odds_data.get("btts", {}),
                    
                    # Raw odds for debugging
                    "raw_odds": m.get("MA", [])[:3] if m.get("MA") else []
                }
                
                matches.append(match_info)
        
        # Cache the results
        _matches_cache = matches
        _cache_timestamp = current_time
        
        logger.info(f"✅ Successfully fetched {len(matches)} matches")
        logger.info(f"📊 Sample odds from first match: {matches[0]['odds_1x2'] if matches else 'No matches'}")
        
        # Save to file for backup
        try:
            with open("prematch_matches.json", "w", encoding="utf-8") as f:
                json.dump(matches, f, ensure_ascii=False, indent=2)
            logger.info("💾 Matches saved to prematch_matches.json")
        except Exception as e:
            logger.warning(f"⚠️ Could not save matches to file: {e}")
        
        return matches
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch matches: {e}", exc_info=True)
        
        # If we have cached data, return it even if stale
        if _matches_cache is not None:
            logger.info("⚠️ Returning stale cached data due to fetch error")
            return _matches_cache
        
        # Return empty list if no data available
        return []

def _format_match_datetime(date_str: str, time_str: str) -> str:
    """
    Format match date and time to ISO format.
    
    Args:
        date_str: Date string in DD.MM.YYYY format
        time_str: Time string in HH:MM format
        
    Returns:
        ISO formatted datetime string
    """
    try:
        if not date_str or not time_str:
            return datetime.now().isoformat()
        
        # Parse DD.MM.YYYY HH:MM
        datetime_str = f"{date_str} {time_str}"
        dt = datetime.strptime(datetime_str, "%d.%m.%Y %H:%M")
        return dt.isoformat()
    except Exception as e:
        logger.warning(f"⚠️ Date parsing error: {e} - Date: {date_str}, Time: {time_str}")
        return datetime.now().isoformat()

def _extract_all_odds(ma_data: List[Dict]) -> Dict[str, Any]:
    """
    Extract all types of odds from MA data.
    
    Args:
        ma_data: List of betting market data
        
    Returns:
        Dictionary with different types of odds
    """
    result = {
        "1x2": {"1": 0.0, "X": 0.0, "2": 0.0},
        "over_under": {"over_2.5": 0.0, "under_2.5": 0.0},
        "btts": {"yes": 0.0, "no": 0.0}
    }
    
    if not ma_data:
        return result
    
    # Process each betting market
    for market in ma_data:
        market_type_id = market.get("MTID", 0)
        oca_list = market.get("OCA", [])
        
        # MTID 1: 1X2 (Maç Sonucu)
        if market_type_id == 1 and len(oca_list) >= 3:
            try:
                result["1x2"]["1"] = float(oca_list[0].get("O", 0) or 0)
                result["1x2"]["X"] = float(oca_list[1].get("O", 0) or 0)
                result["1x2"]["2"] = float(oca_list[2].get("O", 0) or 0)
            except (ValueError, TypeError, IndexError) as e:
                logger.debug(f"Error parsing 1X2 odds: {e}")
        
        # MTID 5: Alt/Üst 2.5 (Over/Under)
        elif market_type_id == 5 and len(oca_list) >= 2:
            try:
                result["over_under"]["over_2.5"] = float(oca_list[0].get("O", 0) or 0)
                result["over_under"]["under_2.5"] = float(oca_list[1].get("O", 0) or 0)
            except (ValueError, TypeError, IndexError) as e:
                logger.debug(f"Error parsing O/U odds: {e}")
        
        # MTID 16: Karşılıklı Gol (BTTS)
        elif market_type_id == 16 and len(oca_list) >= 2:
            try:
                result["btts"]["yes"] = float(oca_list[0].get("O", 0) or 0)
                result["btts"]["no"] = float(oca_list[1].get("O", 0) or 0)
            except (ValueError, TypeError, IndexError) as e:
                logger.debug(f"Error parsing BTTS odds: {e}")
    
    return result

def get_cached_matches() -> List[Dict[str, Any]]:
    """
    Get cached matches if available.
    
    Returns:
        List of cached matches or empty list
    """
    return _matches_cache if _matches_cache is not None else []

def clear_cache():
    """Clear the matches cache."""
    global _matches_cache, _cache_timestamp
    _matches_cache = None
    _cache_timestamp = None
    logger.info("🧹 Matches cache cleared")

def test_fetch():
    """Test function to verify data fetching."""
    print("🧪 Testing Nesine API fetch...")
    matches = fetch_upcoming_matches(force_refresh=True)
    
    if matches:
        print(f"\n✅ Successfully fetched {len(matches)} matches")
        print("\n📋 First match details:")
        first_match = matches[0]
        print(f"  Home: {first_match['home_team']}")
        print(f"  Away: {first_match['away_team']}")
        print(f"  League: {first_match['league_name']}")
        print(f"  Date: {first_match['date']}")
        print(f"  Time: {first_match['time']}")
        print(f"  1X2 Odds: {first_match['odds_1x2']}")
        print(f"  O/U 2.5: {first_match['odds_over_under']}")
        print(f"  BTTS: {first_match['odds_btts']}")
        
        print(f"\n📊 Total matches with valid 1X2 odds: {sum(1 for m in matches if m['odds_1x2']['1'] > 0)}")
    else:
        print("❌ No matches fetched")

if __name__ == "__main__":
    test_fetch()
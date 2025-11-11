import sys, os, logging
from pathlib import Path
from datetime import datetime, timedelta

# ---------- PATH fix ----------
BASE_DIR = Path(__file__).resolve().parent
APP_DIR = BASE_DIR / "app"
if str(BASE_DIR) not in sys.path: sys.path.insert(0, str(BASE_DIR))
if str(APP_DIR) not in sys.path: sys.path.insert(0, str(APP_DIR))

# ---------- MODEL INITIALIZATION ----------
def initialize_model():
    """Modeli kontrol et, yoksa otomatik eğit"""
    print("🔍 Checking model availability...")
    
    model_path = BASE_DIR / "models" / "weighted_model.pkl"
    scaler_path = BASE_DIR / "models" / "weighted_scaler.pkl"
    
    (BASE_DIR / "models").mkdir(exist_ok=True)
    
    if model_path.exists() and scaler_path.exists():
        print("✅ Model files found")
        return True
    else:
        print("🤖 Model not found. Starting automatic training...")
        try:
            from app.weighted_trainer import train_weighted_model
            print("🏋️ Starting model training...")
            success = train_weighted_model()
            
            if success and model_path.exists():
                print("🎉 Model trained successfully!")
                return True
            else:
                print("❌ Model training completed but files not found")
                return False
                
        except Exception as e:
            print(f"💥 Model training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

MODEL_READY = initialize_model()

# ---------- FastAPI ----------
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ---------- Config ----------
try:
    from app.config import MERGED_DATA_FILE, DEFAULT_ODDS
except ImportError:
    MERGED_DATA_FILE = BASE_DIR / "data" / "merged_all.csv"
    DEFAULT_ODDS = {"1": 2.50, "X": 3.20, "2": 2.80}
    print("⚠️  Config not found, using defaults")

# ---------- Nesine fetcher ----------
try:
    from nesine_fetcher_fixed import fetch_upcoming_matches, clear_cache
    NESINE_AVAILABLE = True
    print("✅ Nesine fetcher loaded")
except ImportError as e:
    print(f"⚠️  Nesine fetcher import failed: {e}")
    NESINE_AVAILABLE = False

# ---------- Weighted system ----------
WEIGHTED_AVAILABLE = False
PREDICTOR = None
PREDICTOR_TYPE = "none"

try:
    from app.weighted_predictor import WeightedPredictor
    from app.weighted_feature_engineer import WeightedFeatureEngineer
    
    if MODEL_READY:
        WEIGHTED_AVAILABLE = True
        print("✅ Weighted Prediction System loaded")
    else:
        print("⚠️  Weighted system available but model not ready")
        WEIGHTED_AVAILABLE = False
        
except ImportError as e:
    print(f"⚠️  Weighted system import failed: {e}")
    WEIGHTED_AVAILABLE = False

# ---------- Fallback imports ----------
try:
    from app.predictor import EnhancedEnsemblePredictor
    from app.feature_engineer import MatchFeatureEngineer
    print("✅ Standard system loaded as fallback")
except ImportError as e:
    print(f"⚠️  Standard system import failed: {e}")

# ---------- LiveFeatureEngineer ----------
class LiveFeatureEngineer:
    """Fallback for when dataset is not available"""
    def __init__(self, df=None):
        self.inner = None
        if df is not None:
            try:
                if WEIGHTED_AVAILABLE:
                    self.inner = WeightedFeatureEngineer(df=df)
                    print("✅ WeightedFeatureEngineer initialized with dataset.")
                else:
                    self.inner = MatchFeatureEngineer(df=df, n_jobs=-1)
                    print("✅ MatchFeatureEngineer initialized with dataset.")
            except Exception as e:
                print(f"⚠️  FE init failed, falling back to odds-only: {e}")

    def extract_match_features(self, home_team: str, away_team: str, odds=None, **_):
        if self.inner is not None:
            return self.inner.extract_match_features(home_team, away_team, odds)

        odds = odds or DEFAULT_ODDS
        try:
            o1, ox, o2 = float(odds["1"]), float(odds["X"]), float(odds["2"])
            tot = (1/o1 + 1/ox + 1/o2)
            p1, px, p2 = (1/o1)/tot, (1/ox)/tot, (1/o2)/tot
        except Exception:
            o1, ox, o2, p1, px, p2 = 2.5, 3.2, 2.8, 0.33, 0.33, 0.33

        if WEIGHTED_AVAILABLE:
            return {
                "odds_home": o1, "odds_draw": ox, "odds_away": o2,
                "odds_home_prob": p1, "odds_draw_prob": px, "odds_away_prob": p2,
                "market_margin": tot - 1.0, "market_confidence": 0.5,
                "favorite_odds": min(o1, o2), "underdog_odds": max(o1, o2),
                "odds_spread": abs(o1 - o2),
                "home_value": 0.0, "draw_value": 0.0, "away_value": 0.0,
                "draw_odds_level": 0.5, "draw_market_view": px,
                "clear_favorite": 0.0, "balanced_match": 1.0,
                "h2h_matches": 0, "h2h_home_win_rate": 0.40, "h2h_draw_rate": 0.27,
                "h2h_away_win_rate": 0.33, "h2h_avg_home_goals": 1.3,
                "h2h_avg_away_goals": 1.1, "h2h_avg_total_goals": 2.4,
                "h2h_high_scoring": 0.0, "h2h_draw_tendency": 0.0,
                "home_form_win_rate": 0.40, "home_form_points_per_game": 1.2,
                "home_form_avg_goals_scored": 1.2, "home_form_avg_goals_conceded": 1.2,
                "home_form_momentum": 0.5,
                "away_form_win_rate": 0.40, "away_form_points_per_game": 1.2,
                "away_form_avg_goals_scored": 1.2, "away_form_avg_goals_conceded": 1.2,
                "away_form_momentum": 0.5,
            }
        else:
            return {
                "odds_home": o1, "odds_draw": ox, "odds_away": o2,
                "odds_home_prob": p1, "odds_draw_prob": px, "odds_away_prob": p2,
                "home_strength": 0.0, "away_strength": 0.0,
                "home_attack_power": 1.0, "away_attack_power": 1.0,
                "home_defense_power": 1.0, "away_defense_power": 1.0,
                "home_momentum": 0.0, "away_momentum": 0.0,
                "home_clean_sheet_rate": 0.3, "away_clean_sheet_rate": 0.3,
                "strength_balance": 0.0, "form_balance": 0.0, 
                "goal_difference_balance": 0.0,
                "h2h_draw_rate": 0.0, "xG_balance": 0.0, 
                "poisson_draw_prob": 0.0,
                "both_strong_defense": 0.0, "draw_meta_score": 0.0,
                "match_importance_score": 0.5,
            }

# ---------- Load dataset ----------
def _load_dataset_df():
    import pandas as pd
    if MERGED_DATA_FILE.exists():
        df = pd.read_csv(MERGED_DATA_FILE)
        print(f"📄 Loaded dataset: {MERGED_DATA_FILE} (rows={len(df)})")
        return df
    print("ℹ️  No dataset found; using odds-only features.")
    return None

DATAFRAME = _load_dataset_df()
FEATURE_ENGINEER = LiveFeatureEngineer(df=DATAFRAME)

# ---------- Initialize Predictor ----------
print("\n🎯 Initializing Predictor...")

if WEIGHTED_AVAILABLE:
    try:
        model_dir = BASE_DIR / "models"
        model_path = model_dir / "weighted_model.pkl"
        
        if model_path.exists():
            print(f"✅ Weighted model found: {model_path}")
            PREDICTOR = WeightedPredictor(
                model_dir=model_dir,
                draw_threshold=0.30,
                enable_monitoring=True,
                enable_logic_validation=True
            )
            PREDICTOR_TYPE = "weighted"
            print("✅ Weighted predictor initialized")
        else:
            raise FileNotFoundError("Weighted model not trained")
            
    except Exception as e:
        print(f"⚠️  Weighted predictor failed: {e}")
        WEIGHTED_AVAILABLE = False

if not WEIGHTED_AVAILABLE or PREDICTOR is None:
    print("🔧 Attempting Standard Predictor...")
    try:
        try:
            from app.config import MODEL_PATH
        except ImportError:
            MODEL_PATH = BASE_DIR / "models" / "model.pkl"
        
        if MODEL_PATH.exists():
            PREDICTOR = EnhancedEnsemblePredictor(
                enable_monitoring=True, 
                enable_strategic=False,
                lazy_load=False
            )
            PREDICTOR_TYPE = "standard"
            print("✅ Standard predictor initialized")
        else:
            PREDICTOR_TYPE = "no_model"
    except Exception as e:
        print(f"❌ Standard predictor failed: {e}")
        PREDICTOR_TYPE = "no_model"

# ---------- FastAPI App ----------
app = FastAPI(
    title="Predicta Weighted API",
    description="Football match prediction (75% Odds, 15% H2H, 10% Form)",
    version="3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

WEB_DIR = BASE_DIR / "web"
if WEB_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(WEB_DIR), html=True), name="ui")

MATCHES_CACHE = {"data": [], "timestamp": None, "ttl_minutes": 5}

# ---------- Helper Functions ----------
def _convert_nesine_to_api_format(nesine_matches):
    """Convert Nesine format to API format"""
    api_matches = []
    for m in nesine_matches:
        odds_1x2 = m.get("odds_1x2", {})
        if not odds_1x2.get("1") or odds_1x2.get("1") == 0:
            continue
        
        api_matches.append({
            "match_id": m.get("match_id", ""),
            "date": m.get("date", ""),
            "league_code": m.get("league_code", ""),
            "league_name": m.get("league_name", "Unknown League"),
            "home_team": m.get("home_team", "Unknown Home"),
            "away_team": m.get("away_team", "Unknown Away"),
            "odds": {
                "1": odds_1x2.get("1", 0.0),
                "X": odds_1x2.get("X", 0.0),
                "2": odds_1x2.get("2", 0.0)
            },
            "is_live": m.get("is_live", False),
            "odds_over_under": m.get("odds_over_under", {}),
            "odds_btts": m.get("odds_btts", {}),
        })
    return api_matches

def _dummy_upcoming():
    """Dummy matches for testing"""
    now = datetime.now()
    teams = [
        ("Barcelona", "Real Madrid"),
        ("Fenerbahçe", "Galatasaray"),
        ("Beşiktaş", "Trabzonspor"),
    ]
    out = []
    for i, (h, a) in enumerate(teams, 1):
        out.append({
            "match_id": f"dummy_{i}",
            "date": (now + timedelta(days=i)).isoformat(),
            "league_code": "TEST",
            "league_name": "Test League",
            "home_team": h, 
            "away_team": a,
            "odds": {
                "1": 1.80 + i*0.1, 
                "X": 3.10 + i*0.05, 
                "2": 2.40 + i*0.08
            }
        })
    return out

# ---------- API Endpoints ----------
@app.get("/")
def root():
    return {
        "message": f"Predicta API - {PREDICTOR_TYPE.upper()}",
        "version": "3.0",
        "predictor_type": PREDICTOR_TYPE,
        "weighted_system": WEIGHTED_AVAILABLE,
        "model_loaded": PREDICTOR is not None,
        "endpoints": {
            "docs": "/docs",
            "status": "/api/status",
            "upcoming": "/api/matches/upcoming",
            "predict_match": "/api/predict/match",
            "predict_ou": "/api/predict/over-under",
            "predict_btts": "/api/predict/btts",
            "full_match": "/api/predict/full-match",
            "validate": "/api/validate/full-prediction"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if MODEL_READY else "model_missing",
        "timestamp": datetime.now().isoformat(),
        "model_ready": MODEL_READY
    }

@app.get("/api/status")
def api_status():
    return {
        "status": "running",
        "predictor_type": PREDICTOR_TYPE,
        "weighted_system": WEIGHTED_AVAILABLE,
        "nesine_fetcher": NESINE_AVAILABLE,
        "model_loaded": PREDICTOR is not None,
    }

@app.get("/api/matches/upcoming")
def get_upcoming(force_refresh: bool = Query(False)):
    """Get upcoming matches"""
    now = datetime.now()
    
    if (not force_refresh and MATCHES_CACHE["data"] and MATCHES_CACHE["timestamp"]
        and (now - MATCHES_CACHE["timestamp"]).total_seconds() < MATCHES_CACHE["ttl_minutes"]*60):
        return {
            "success": True, 
            "cached": True,
            "count": len(MATCHES_CACHE["data"]),
            "matches": MATCHES_CACHE["data"], 
            "timestamp": MATCHES_CACHE["timestamp"].isoformat()
        }
    
    if NESINE_AVAILABLE:
        try:
            nesine_matches = fetch_upcoming_matches(force_refresh=True)
            if nesine_matches:
                api_matches = _convert_nesine_to_api_format(nesine_matches)
                if api_matches:
                    MATCHES_CACHE["data"] = api_matches
                    MATCHES_CACHE["timestamp"] = now
                    return {
                        "success": True, 
                        "cached": False,
                        "source": "nesine",
                        "count": len(api_matches), 
                        "matches": api_matches, 
                        "timestamp": now.isoformat()
                    }
        except Exception as e:
            logging.error(f"Nesine fetch failed: {e}")
    
    dummy_data = _dummy_upcoming()
    MATCHES_CACHE["data"] = dummy_data
    MATCHES_CACHE["timestamp"] = now
    
    return {
        "success": True, 
        "cached": False,
        "source": "dummy",
        "count": len(dummy_data), 
        "matches": dummy_data, 
        "timestamp": now.isoformat()
    }

@app.post("/api/predict/match")
def predict_match(data: dict):
    """Predict 1X2"""
    if PREDICTOR is None or PREDICTOR_TYPE == "no_model":
        return JSONResponse(status_code=503, content={"error": "No model"})
    
    try:
        pred = PREDICTOR.predict_match(
            home_team=data.get("home_team"), 
            away_team=data.get("away_team"),
            odds=data.get("odds"), 
            feature_engineer=FEATURE_ENGINEER
        )
        pred["predictor_type"] = PREDICTOR_TYPE
        return pred
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/predict/over-under")
def predict_over_under(data: dict):
    """Predict Over/Under 2.5"""
    if PREDICTOR is None:
        return JSONResponse(status_code=503, content={"error": "No model"})
    
    try:
        features = FEATURE_ENGINEER.extract_match_features(
            home_team=data.get("home_team"),
            away_team=data.get("away_team"),
            odds=data.get("odds", {})
        )
        
        home_scored = features.get("home_form_avg_goals_scored", 1.2)
        away_scored = features.get("away_form_avg_goals_scored", 1.2)
        home_conceded = features.get("home_form_avg_goals_conceded", 1.2)
        away_conceded = features.get("away_form_avg_goals_conceded", 1.2)
        h2h_total = features.get("h2h_avg_total_goals", 2.4)
        
        expected_home = (home_scored + away_conceded) / 2
        expected_away = (away_scored + home_conceded) / 2
        expected_total = (expected_home + expected_away) * 0.7 + h2h_total * 0.3
        
        if expected_total > 2.75:
            over_prob = 0.75
        elif expected_total > 2.5:
            over_prob = 0.60
        elif expected_total > 2.25:
            over_prob = 0.50
        else:
            over_prob = 0.35
        
        under_prob = 1.0 - over_prob
        prediction = "over" if over_prob > 0.5 else "under"
        
        return {
            "home_team": data.get("home_team"),
            "away_team": data.get("away_team"),
            "prediction": prediction,
            "confidence": max(over_prob, under_prob),
            "expected_goals": round(expected_total, 2),
            "probabilities": {"over": round(over_prob, 4), "under": round(under_prob, 4)},
            "threshold": 2.5
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/predict/btts")
def predict_btts(data: dict):
    """Predict BTTS"""
    if PREDICTOR is None:
        return JSONResponse(status_code=503, content={"error": "No model"})
    
    try:
        features = FEATURE_ENGINEER.extract_match_features(
            home_team=data.get("home_team"),
            away_team=data.get("away_team"),
            odds=data.get("odds", {})
        )
        
        home_attack = features.get("home_form_avg_goals_scored", 1.2)
        away_attack = features.get("away_form_avg_goals_scored", 1.2)
        
        home_prob = min(0.9, home_attack / 1.5)
        away_prob = min(0.9, away_attack / 1.5)
        btts_prob = home_prob * away_prob
        btts_prob = min(0.85, max(0.15, btts_prob))
        
        prediction = btts_prob > 0.5
        
        return {
            "home_team": data.get("home_team"),
            "away_team": data.get("away_team"),
            "prediction": prediction,
            "prediction_label": "KG Var" if prediction else "KG Yok",
            "confidence": max(btts_prob, 1.0 - btts_prob),
            "probabilities": {"yes": round(btts_prob, 4), "no": round(1.0 - btts_prob, 4)}
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/predict/full-match")
async def predict_full_match(data: dict):
    """Full prediction"""
    if PREDICTOR is None:
        return JSONResponse(status_code=503, content={"error": "No model"})
    
    try:
        ms = predict_match(data)
        ou = predict_over_under(data)
        btts = predict_btts(data)
        
        return {
            "match": {
                "home_team": data.get("home_team"),
                "away_team": data.get("away_team")
            },
            "predictions": {
                "match_result": ms,
                "over_under": ou,
                "btts": btts
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/validate/full-prediction")
async def validate_full_prediction(data: dict):
    """Validate predictions"""
    if PREDICTOR is None or not hasattr(PREDICTOR, 'validate_full_prediction_set'):
        return {"is_valid": True, "warnings": []}
    
    try:
        result = PREDICTOR.validate_full_prediction_set(
            ms_prediction=data.get("ms_prediction"),
            ms_confidence=float(data.get("ms_confidence", 0.0)),
            ou_prediction=data.get("ou_prediction"),
            ou_confidence=float(data.get("ou_confidence", 0.0)),
            btts_prediction=data.get("btts_prediction"),
            btts_confidence=float(data.get("btts_confidence", 0.0))
        )
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print("🚀 Predicta API Starting")
    print(f"🎯 Type: {PREDICTOR_TYPE.upper()}")
    print(f"🤖 Model: {'LOADED' if PREDICTOR else 'MISSING'}")
    print("=" * 60)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

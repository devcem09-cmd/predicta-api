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
    
    # Models klasörünü oluştur
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

# Modeli başlangıçta yükle
MODEL_READY = initialize_model()

# ---------- FastAPI ----------
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ---------- Try to import config ----------
try:
    from app.config import MERGED_DATA_FILE, DEFAULT_ODDS
except ImportError:
    MERGED_DATA_FILE = BASE_DIR / "data" / "merged_all.csv"
    DEFAULT_ODDS = {"1": 2.50, "X": 3.20, "2": 2.80}
    print("⚠️  Config not found, using defaults")

# ⭐ NEW: Import Nesine fetcher
try:
    from nesine_fetcher_fixed import fetch_upcoming_matches, clear_cache
    NESINE_AVAILABLE = True
    print("✅ Nesine fetcher loaded")
except ImportError as e:
    print(f"⚠️  Nesine fetcher import failed: {e}")
    NESINE_AVAILABLE = False

# ⭐ NEW: Weighted system import
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

# Fallback imports
try:
    from app.predictor import EnhancedEnsemblePredictor
    from app.feature_engineer import MatchFeatureEngineer
    print("✅ Standard system loaded as fallback")
except ImportError as e:
    print(f"⚠️  Standard system import failed: {e}")

# ---------- FastAPI App Creation ----------
app = FastAPI(title="Predicta Weighted API", version="2.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory=BASE_DIR / "web"), name="static")

# ---------- API Routes ----------
@app.get("/")
async def root():
    return {
        "message": "Predicta Weighted API", 
        "version": "2.0",
        "model_ready": MODEL_READY,
        "weighted_system": WEIGHTED_AVAILABLE,
        "nesine_available": NESINE_AVAILABLE
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Render"""
    return {
        "status": "healthy" if MODEL_READY else "model_missing",
        "timestamp": datetime.now().isoformat(),
        "model_ready": MODEL_READY
    }

# ---------- Lightweight live FE ----------
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

        # Minimal fallback features
        odds = odds or DEFAULT_ODDS
        try:
            o1, ox, o2 = float(odds["1"]), float(odds["X"]), float(odds["2"])
            tot = (1/o1 + 1/ox + 1/o2)
            p1, px, p2 = (1/o1)/tot, (1/ox)/tot, (1/o2)/tot
        except Exception:
            o1, ox, o2, p1, px, p2 = 2.5, 3.2, 2.8, 0.33, 0.33, 0.33

        # Weighted system expected features
        if WEIGHTED_AVAILABLE:
            return {
                # Odds features (18)
                "odds_home": o1, "odds_draw": ox, "odds_away": o2,
                "odds_home_prob": p1, "odds_draw_prob": px, "odds_away_prob": p2,
                "market_margin": tot - 1.0,
                "market_confidence": 0.5,
                "favorite_odds": min(o1, o2),
                "underdog_odds": max(o1, o2),
                "odds_spread": abs(o1 - o2),
                "home_value": 0.0, "draw_value": 0.0, "away_value": 0.0,
                "draw_odds_level": 0.5,
                "draw_market_view": px,
                "clear_favorite": 0.0,
                "balanced_match": 1.0,
                
                # H2H features (9) - defaults
                "h2h_matches": 0, "h2h_home_win_rate": 0.40, "h2h_draw_rate": 0.27,
                "h2h_away_win_rate": 0.33, "h2h_avg_home_goals": 1.3,
                "h2h_avg_away_goals": 1.1, "h2h_avg_total_goals": 2.4,
                "h2h_high_scoring": 0.0, "h2h_draw_tendency": 0.0,
                
                # Form features (10) - defaults
                "home_form_win_rate": 0.40, "home_form_points_per_game": 1.2,
                "home_form_avg_goals_scored": 1.2, "home_form_avg_goals_conceded": 1.2,
                "home_form_momentum": 0.5,
                "away_form_win_rate": 0.40, "away_form_points_per_game": 1.2,
                "away_form_avg_goals_scored": 1.2, "away_form_avg_goals_conceded": 1.2,
                "away_form_momentum": 0.5,
            }
        else:
            # Fallback for standard system
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
print("\n🎯 Initializing Weighted Predictor...")

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
            print("✅ Weighted predictor initialized successfully")
        else:
            print(f"⚠️  Weighted model not found at {model_path}")
            print("💡 Train the model first: python app/weighted_trainer.py")
            raise FileNotFoundError("Weighted model not trained yet")
            
    except Exception as e:
        print(f"⚠️  Weighted predictor failed, falling back to standard: {e}")
        WEIGHTED_AVAILABLE = False

# Fallback to standard system
if not WEIGHTED_AVAILABLE or PREDICTOR is None:
    print("🔧 Attempting to initialize Standard Predictor...")
    try:
        try:
            from app.config import MODEL_PATH
        except ImportError:
            MODEL_PATH = BASE_DIR / "models" / "model.pkl"
        
        if MODEL_PATH.exists():
            print(f"✅ Standard model found: {MODEL_PATH}")
            PREDICTOR = EnhancedEnsemblePredictor(
                enable_monitoring=True, 
                enable_strategic=False,
                lazy_load=False
            )
            PREDICTOR_TYPE = "standard"
            print("✅ Standard predictor initialized successfully")
        else:
            print(f"⚠️  No model found at {MODEL_PATH}")
            print("💡 Train a model first:")
            print("   - Weighted: python app/weighted_trainer.py")
            print("   - Standard: python train_real_model.py")
            print("🚫 API will start but predictions will fail until model is trained")
            PREDICTOR_TYPE = "no_model"
    except Exception as e:
        print(f"❌ Standard predictor failed: {e}")
        print("🚫 No predictor available - API will have limited functionality")
        PREDICTOR_TYPE = "no_model"

# ---------- FastAPI App ----------
app = FastAPI(
    title="Predicta Weighted API",
    description="Football match prediction with weighted feature system (75% Odds, 15% H2H, 10% Form)",
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


@app.get("/api/status")
def api_status():
    """API status and model information"""
    status = {
        "status": "running",
        "predictor_type": PREDICTOR_TYPE,
        "weighted_system": WEIGHTED_AVAILABLE,
        "nesine_fetcher": NESINE_AVAILABLE,
        "web_ui_available": WEB_DIR.exists(),
        "dataset_loaded": DATAFRAME is not None,
        "model_loaded": PREDICTOR is not None,
    }
    
    if PREDICTOR_TYPE == "no_model":
        status["warning"] = "No trained model available"
        status["instructions"] = {
            "weighted": "python app/weighted_trainer.py",
            "standard": "python train_real_model.py"
        }
    elif PREDICTOR_TYPE == "weighted":
        weighted_model_path = BASE_DIR / "models" / "weighted_model.pkl"
        status["model_path"] = str(weighted_model_path)
        status["model_trained"] = weighted_model_path.exists()
        status["feature_priorities"] = {
            "odds": "75%",
            "h2h": "15%",
            "form": "10%"
        }
    else:
        try:
            from app.config import MODEL_PATH
        except ImportError:
            MODEL_PATH = BASE_DIR / "models" / "model.pkl"
        status["model_path"] = str(MODEL_PATH)
        status["model_trained"] = MODEL_PATH.exists()
        if PREDICTOR and hasattr(PREDICTOR, 'best_models_per_class'):
            status["smart_selection_enabled"] = bool(PREDICTOR.best_models_per_class)
    
    return status


def _convert_nesine_to_api_format(nesine_matches):
    """Convert Nesine format to API format"""
    api_matches = []
    
    for m in nesine_matches:
        # Extract 1X2 odds
        odds_1x2 = m.get("odds_1x2", {})
        
        # Skip matches without valid odds
        if not odds_1x2.get("1") or odds_1x2.get("1") == 0:
            continue
        
        api_match = {
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
            
            # Additional odds if available
            "odds_over_under": m.get("odds_over_under", {}),
            "odds_btts": m.get("odds_btts", {}),
        }
        
        api_matches.append(api_match)
    
    return api_matches


@app.get("/api/matches/upcoming")
def get_upcoming(force_refresh: bool = Query(False)):
    """Get upcoming matches from Nesine (cached)"""
    now = datetime.now()
    
    # Check cache
    if (not force_refresh and MATCHES_CACHE["data"] and MATCHES_CACHE["timestamp"]
        and (now - MATCHES_CACHE["timestamp"]).total_seconds() < MATCHES_CACHE["ttl_minutes"]*60):
        return {
            "success": True, 
            "cached": True, 
            "source": "nesine_cached" if NESINE_AVAILABLE else "dummy_cached",
            "count": len(MATCHES_CACHE["data"]),
            "matches": MATCHES_CACHE["data"], 
            "timestamp": MATCHES_CACHE["timestamp"].isoformat()
        }
    
    # Fetch fresh data
    if NESINE_AVAILABLE:
        try:
            print("🌐 Fetching matches from Nesine API...")
            nesine_matches = fetch_upcoming_matches(force_refresh=True)
            
            if nesine_matches:
                # Convert to API format
                api_matches = _convert_nesine_to_api_format(nesine_matches)
                
                if api_matches:
                    MATCHES_CACHE["data"] = api_matches
                    MATCHES_CACHE["timestamp"] = now
                    
                    print(f"✅ Loaded {len(api_matches)} matches from Nesine")
                    
                    return {
                        "success": True, 
                        "cached": False,
                        "source": "nesine",
                        "count": len(api_matches), 
                        "matches": api_matches, 
                        "timestamp": now.isoformat()
                    }
                else:
                    print("⚠️  No valid matches with odds found")
            else:
                print("⚠️  Nesine returned empty data")
                
        except Exception as e:
            print(f"❌ Nesine fetch error: {e}")
            logging.error(f"Nesine fetch failed: {e}", exc_info=True)
    
    # Fallback to dummy data
    print("ℹ️  Using dummy test data")
    dummy_data = _dummy_upcoming()
    MATCHES_CACHE["data"] = dummy_data
    MATCHES_CACHE["timestamp"] = now
    
    return {
        "success": True, 
        "cached": False,
        "source": "dummy_fallback",
        "count": len(dummy_data), 
        "matches": dummy_data, 
        "timestamp": now.isoformat(),
        "warning": "Using dummy data. Nesine fetcher may not be working."
    }


def _dummy_upcoming():
    """Dummy matches for testing"""
    now = datetime.now()
    teams = [
        ("Barcelona", "Real Madrid"),
        ("Fenerbahçe", "Galatasaray"),
        ("Beşiktaş", "Trabzonspor"),
        ("Bayern Munich", "Borussia Dortmund"),
        ("Man United", "Liverpool"),
        ("Arsenal", "Chelsea"),
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


@app.get("/api/matches/upcoming/predicted")
def get_predicted_upcoming(force_refresh: bool = Query(False)):
    """Get predicted upcoming matches"""
    resp = get_upcoming(force_refresh)
    if not resp.get("success"): 
        return resp
    
    if PREDICTOR is None or PREDICTOR_TYPE == "no_model":
        return JSONResponse(
            status_code=503,
            content={
                "error": "No trained model available",
                "message": "Please train a model first",
                "instructions": {
                    "weighted": "python app/weighted_trainer.py",
                    "standard": "python train_real_model.py"
                }
            }
        )
    
    prediction_summary = {
        "total": len(resp["matches"]),
        "predictor_type": PREDICTOR_TYPE,
        "weighted_system": WEIGHTED_AVAILABLE,
        "data_source": resp.get("source", "unknown")
    }
    
    if PREDICTOR_TYPE == "weighted":
        prediction_summary["feature_priorities"] = {
            "odds": "75%",
            "h2h": "15%",
            "form": "10%"
        }
    
    for m in resp["matches"]:
        try:
            pred = PREDICTOR.predict_match(
                home_team=m["home_team"], 
                away_team=m["away_team"],
                odds=m.get("odds"), 
                feature_engineer=FEATURE_ENGINEER
            )
            
            prediction_result = {
                "prediction": pred.get("prediction"),
                "prediction_name": pred.get("prediction_name"),
                "confidence": pred.get("confidence", 0.0),
                "probabilities": pred.get("probabilities", {}),
                "method": pred.get("prediction_method", "unknown") if PREDICTOR_TYPE == "standard" else "weighted",
            }
            
            if PREDICTOR_TYPE == "weighted":
                prediction_result["odds_confidence"] = pred.get("odds_confidence", 0.0)
                prediction_result["feature_priorities"] = pred.get("feature_priorities", {})
                if "odds_analysis" in pred:
                    prediction_result["odds_analysis"] = pred["odds_analysis"]
            
            elif PREDICTOR_TYPE == "standard" and "models_used" in pred:
                pred_label = pred.get("prediction", "?")
                if pred_label == "1":
                    model_used = pred["models_used"].get("Home Win", "ensemble")
                elif pred_label == "X":
                    model_used = pred["models_used"].get("Draw", "ensemble")
                elif pred_label == "2":
                    model_used = pred["models_used"].get("Away Win", "ensemble")
                else:
                    model_used = "unknown"
                
                prediction_result["model_used"] = model_used
                prediction_result["all_models"] = pred.get("models_used", {})
            
            m["prediction"] = prediction_result
            
        except Exception as e:
            logging.error(f"Prediction error for {m['match_id']}: {e}")
            m["prediction"] = {
                "prediction": "?", 
                "confidence": 0.0, 
                "error": str(e),
                "method": "error"
            }
    
    return {
        "success": True, 
        "cached": resp.get("cached", False),
        "source": resp.get("source", "unknown"),
        "count": len(resp["matches"]), 
        "matches": resp["matches"], 
        "timestamp": resp.get("timestamp"),
        "prediction_summary": prediction_summary
    }


@app.post("/api/predict/match")
def predict_match(data: dict):
    """Predict single match"""
    
    if PREDICTOR is None or PREDICTOR_TYPE == "no_model":
        return JSONResponse(
            status_code=503,
            content={
                "error": "No trained model available",
                "message": "Please train a model first",
                "instructions": {
                    "weighted": "python app/weighted_trainer.py",
                    "standard": "python train_real_model.py"
                }
            }
        )
    
    try:
        pred = PREDICTOR.predict_match(
            home_team=data.get("home_team"), 
            away_team=data.get("away_team"),
            odds=data.get("odds"), 
            feature_engineer=FEATURE_ENGINEER
        )
        
        pred["predictor_type"] = PREDICTOR_TYPE
        pred["weighted_system"] = WEIGHTED_AVAILABLE
        
        return pred
        
    except Exception as e:
        logging.exception("Prediction failed")
        return JSONResponse(
            status_code=500, 
            content={
                "error": str(e), 
                "message": "Prediction failed",
                "predictor_type": PREDICTOR_TYPE
            }
        )


# ⭐ YENİ: Alt/Üst 2.5 Tahmin Endpoint'i
@app.post("/api/predict/over-under")
def predict_over_under(data: dict):
    """Predict Over/Under 2.5 goals"""
    
    if PREDICTOR is None or PREDICTOR_TYPE == "no_model":
        return JSONResponse(
            status_code=503,
            content={
                "error": "No trained model available",
                "message": "Please train a model first"
            }
        )
    
    try:
        # Feature extraction
        features = FEATURE_ENGINEER.extract_match_features(
            home_team=data.get("home_team"),
            away_team=data.get("away_team"),
            odds=data.get("odds")
        )
        
        # Beklenen gol hesaplama
        home_goals_scored = features.get("home_form_avg_goals_scored", 1.2)
        away_goals_scored = features.get("away_form_avg_goals_scored", 1.2)
        h2h_avg_total = features.get("h2h_avg_total_goals", 2.4)
        
        # Weighted average
        expected_goals = (home_goals_scored * 0.4 + 
                         away_goals_scored * 0.4 + 
                         h2h_avg_total * 0.2)
        
        # Probability calculation
        if expected_goals > 2.75:
            over_prob = 0.70
        elif expected_goals > 2.5:
            over_prob = 0.60
        elif expected_goals > 2.25:
            over_prob = 0.50
        else:
            over_prob = 0.35
        
        under_prob = 1.0 - over_prob
        
        prediction = "over" if over_prob > 0.5 else "under"
        confidence = max(over_prob, under_prob)
        
        return {
            "home_team": data.get("home_team"),
            "away_team": data.get("away_team"),
            "prediction": prediction,
            "confidence": confidence,
            "expected_goals": round(expected_goals, 2),
            "probabilities": {
                "over": round(over_prob, 4),
                "under": round(under_prob, 4)
            },
            "threshold": 2.5
        }
        
    except Exception as e:
        logging.exception("Over/Under prediction failed")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "message": "Over/Under prediction failed"}
        )


# ⭐ YENİ: KG Var/Yok (Both Teams To Score) Tahmin Endpoint'i
@app.post("/api/predict/btts")
def predict_btts(data: dict):
    """Predict Both Teams To Score (Karşılıklı Gol)"""
    
    if PREDICTOR is None or PREDICTOR_TYPE == "no_model":
        return JSONResponse(
            status_code=503,
            content={
                "error": "No trained model available",
                "message": "Please train a model first"
            }
        )
    
    try:
        # Feature extraction
        features = FEATURE_ENGINEER.extract_match_features(
            home_team=data.get("home_team"),
            away_team=data.get("away_team"),
            odds=data.get("odds")
        )
        
        # Attack and defense strengths
        home_attack = features.get("home_form_avg_goals_scored", 1.2)
        away_attack = features.get("away_form_avg_goals_scored", 1.2)
        home_defense = features.get("home_form_avg_goals_conceded", 1.2)
        away_defense = features.get("away_form_avg_goals_conceded", 1.2)
        
        # BTTS probability calculation
        # Both teams need good attack
        btts_score = (
            (home_attack / 1.5) * 0.25 +
            (away_attack / 1.5) * 0.25 +
            (home_defense / 1.5) * 0.25 +
            (away_defense / 1.5) * 0.25
        )
        
        btts_prob = min(0.85, max(0.15, btts_score))
        no_btts_prob = 1.0 - btts_prob
        
        prediction = btts_prob > 0.5
        confidence = max(btts_prob, no_btts_prob)
        
        return {
            "home_team": data.get("home_team"),
            "away_team": data.get("away_team"),
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": {
                "yes": round(btts_prob, 4),
                "no": round(no_btts_prob, 4)
            },
            "factors": {
                "home_attack_strength": round(home_attack, 2),
                "away_attack_strength": round(away_attack, 2),
                "home_defense_weakness": round(home_defense, 2),
                "away_defense_weakness": round(away_defense, 2)
            }
        }
        
    except Exception as e:
        logging.exception("BTTS prediction failed")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "message": "BTTS prediction failed"}
        )

@app.post("/api/validate/full-prediction")
async def validate_full_prediction(
    ms_prediction: str,
    ms_confidence: float,
    ou_prediction: str,
    ou_confidence: float,
    btts_prediction: bool,
    btts_confidence: float
):
    """3 tahmini birlikte kontrol et"""
    
    # ❌ HATALI: predictor
    # ✅ DOĞRU: PREDICTOR (büyük harf)
    
    if PREDICTOR is None or PREDICTOR_TYPE == "no_model":
        return JSONResponse(
            status_code=503,
            content={
                "error": "No trained model available",
                "message": "Please train a model first"
            }
        )
    
    try:
        result = PREDICTOR.validate_full_prediction_set(  # ✅ Düzeltildi
            ms_prediction=ms_prediction,
            ms_confidence=ms_confidence,
            ou_prediction=ou_prediction,
            ou_confidence=ou_confidence,
            btts_prediction=btts_prediction,
            btts_confidence=btts_confidence
        )
        return result
    except Exception as e:
        logging.exception("Validation failed")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "message": "Validation failed"}
        )
@app.post("/api/matches/clear-cache")
def clear_matches_cache():
    """Clear matches cache"""
    global MATCHES_CACHE
    MATCHES_CACHE = {"data": [], "timestamp": None, "ttl_minutes": 5}
    
    if NESINE_AVAILABLE:
        clear_cache()
    
    return {
        "success": True,
        "message": "Cache cleared"
    }


@app.get("/api/model/info")
def get_model_info():
    """Get model information"""
    try:
        info = {
            "predictor_type": PREDICTOR_TYPE,
            "weighted_system": WEIGHTED_AVAILABLE,
            "nesine_fetcher": NESINE_AVAILABLE,
        }
        
        if PREDICTOR_TYPE == "weighted" and PREDICTOR:
            info["feature_priorities"] = {
                "odds": "75%",
                "h2h": "15%",
                "form": "10%"
            }
            
            if hasattr(PREDICTOR, 'metadata') and PREDICTOR.metadata:
                info["training_info"] = {
                    "training_date": PREDICTOR.metadata.get("training_date"),
                    "test_accuracy": PREDICTOR.metadata.get("test_accuracy"),
                    "test_logloss": PREDICTOR.metadata.get("test_logloss"),
                    "draw_recall": PREDICTOR.metadata.get("draw_recall"),
                    "draw_precision": PREDICTOR.metadata.get("draw_precision"),
                    "n_features": PREDICTOR.metadata.get("n_features"),
                }
        
        elif PREDICTOR_TYPE == "standard" and PREDICTOR:
            info["models_loaded"] = getattr(PREDICTOR, '_models_loaded', False)
            
            if hasattr(PREDICTOR, 'best_models_per_class'):
                info["smart_selection_enabled"] = bool(PREDICTOR.best_models_per_class)
        
        return info
        
    except Exception as e:
        logging.exception("Model info retrieval failed")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/api/performance/stats")
def get_performance_stats():
    """Get performance statistics"""
    if PREDICTOR is None:
        return JSONResponse(
            status_code=503,
            content={"error": "No predictor loaded"}
        )
    
    try:
        stats = PREDICTOR.get_performance_stats()
        
        return {
            "success": True,
            "stats": stats,
            "predictor_type": PREDICTOR_TYPE,
            "weighted_system": WEIGHTED_AVAILABLE
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/")
def root():
    """API information"""
    features = [
        "⚖️  Weighted Feature System (75% Odds, 15% H2H, 10% Form)" if WEIGHTED_AVAILABLE else "🧠 Smart Model Selection",
        "🌐 Nesine.com Live Data Integration" if NESINE_AVAILABLE else "📊 Manual Data Input",
        "📊 Advanced Features",
        "🤖 Ensemble Learning",
        "⚡ Real-time Predictions",
        "🎯 High Accuracy",
        "📈 Performance Monitoring"
    ]
    
    return {
        "message": f"Predicta API - {PREDICTOR_TYPE.upper()} Mode ✅",
        "version": "3.0",
        "description": "Advanced football match prediction with Nesine.com integration",
        "predictor_type": PREDICTOR_TYPE,
        "weighted_system": WEIGHTED_AVAILABLE,
        "nesine_fetcher": NESINE_AVAILABLE,
        "model_loaded": PREDICTOR is not None,
        "features": features,
        "endpoints": {
            "api_docs": "/docs",
            "status": "/api/status",
            "model_info": "/api/model/info",
            "performance_stats": "/api/performance/stats",
            "upcoming_matches": "/api/matches/upcoming",
            "predicted_matches": "/api/matches/upcoming/predicted",
            "predict_single": "/api/predict/match [POST]",
            "predict_over_under": "/api/predict/over-under [POST]",
            "predict_btts": "/api/predict/btts [POST]",
            "clear_cache": "/api/matches/clear-cache [POST]"
        }
    }


@app.on_event("startup")
async def startup_event():
    """Runs on API startup"""
    print("=" * 60)
    print("🚀 Predicta API Starting...")
    print("=" * 60)
    print(f"📁 Base Directory: {BASE_DIR}")
    print(f"🎯 Predictor Type: {PREDICTOR_TYPE.upper()}")
    print(f"⚖️  Weighted System: {'ENABLED' if WEIGHTED_AVAILABLE else 'DISABLED'}")
    print(f"🌐 Nesine Fetcher: {'ENABLED ✅' if NESINE_AVAILABLE else 'DISABLED ⚠️'}")
    print(f"🤖 Model Status: {'LOADED ✅' if PREDICTOR else 'NOT LOADED ⚠️'}")
    print(f"🌐 Web UI: {'Available' if WEB_DIR.exists() else 'Not Available'}")
    print(f"📊 Dataset: {'Loaded' if DATAFRAME is not None else 'Not Loaded'}")
    
    if not NESINE_AVAILABLE:
        print(f"⚠️  WARNING: Nesine fetcher not available!")
        print(f"   Using dummy test data instead of real matches")
    
    if PREDICTOR_TYPE == "no_model":
        print(f"⚠️  WARNING: No trained model found!")
        print(f"💡 Train a model:")
        print(f"   Weighted: python app/weighted_trainer.py")
        print(f"   Standard: python train_real_model.py")
    elif PREDICTOR_TYPE == "weighted":
        print(f"📈 Feature Priorities: 75% Odds | 15% H2H | 10% Form")
        print(f"🎯 Predictions: 1X2 + Over/Under 2.5 + BTTS")
    elif PREDICTOR_TYPE == "standard" and PREDICTOR and hasattr(PREDICTOR, 'best_models_per_class'):
        if PREDICTOR.best_models_per_class:
            print(f"🧠 Smart Selection: ACTIVE")
    
    print("=" * 60)


if __name__ == "__main__":
    import uvicorn
    import os
    
    print(f"🎯 Starting Predicta Weighted API...")
    print(f"📊 Model Status: {'READY' if MODEL_READY else 'NOT READY'}")
    print(f"⚖️ Weighted System: {'AVAILABLE' if WEIGHTED_AVAILABLE else 'UNAVAILABLE'}")
    
    # Render PORT'u al veya default 8000 kullan
    port = int(os.environ.get("PORT", 8000))
    print(f"🌐 Binding to port: {port}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0",  # Tüm network interface'leri dinle
        port=port,        # Render'ın verdiği PORT'u kullan
        log_level="info"
    )

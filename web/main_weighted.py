import sys, os, logging
from pathlib import Path
from datetime import datetime, timedelta

# ---------- PATH fix ----------
BASE_DIR = Path(__file__).resolve().parent

# ⭐ FIX: Check if we're in 'main' subdirectory
if BASE_DIR.name == 'main':
    # We're inside main/ folder, stay here
    PROJECT_ROOT = BASE_DIR
else:
    # We're at root, check if main/ exists
    if (BASE_DIR / 'main').exists():
        PROJECT_ROOT = BASE_DIR / 'main'
    else:
        PROJECT_ROOT = BASE_DIR

print(f"📁 Project Root: {PROJECT_ROOT}")

APP_DIR = PROJECT_ROOT / "app"
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR) not in sys.path: sys.path.insert(0, str(APP_DIR))

# ---------- FastAPI ----------
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ---------- Try to import config ----------
try:
    from app.config import MERGED_DATA_FILE, DEFAULT_ODDS
except ImportError:
    MERGED_DATA_FILE = PROJECT_ROOT / "data" / "merged_all.csv"
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
    WEIGHTED_AVAILABLE = True
    print("✅ Weighted Prediction System loaded")
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

        # Minimal fallback features (same as before)
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
print("\n🎯 Initializing Weighted Predictor...")

if WEIGHTED_AVAILABLE:
    try:
        model_dir = PROJECT_ROOT / "models"
        model_path = model_dir / "weighted_model.pkl"
        
        # ⭐ DEBUG: Print all possible model paths
        print(f"🔍 Looking for model at: {model_path}")
        print(f"🔍 Model exists: {model_path.exists()}")
        print(f"🔍 Model dir contents: {list(model_dir.glob('*.pkl')) if model_dir.exists() else 'dir not found'}")
        
        if model_path.exists():
            print(f"✅ Weighted model found: {model_path}")
            PREDICTOR = WeightedPredictor(
                model_dir=model_dir,
                draw_threshold=0.30,
                enable_monitoring=True
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

# Fallback to standard system (rest of the code remains the same)
if not WEIGHTED_AVAILABLE or PREDICTOR is None:
    print("🔧 Attempting to initialize Standard Predictor...")
    try:
        try:
            from app.config import MODEL_PATH
        except ImportError:
            MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
        
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

WEB_DIR = PROJECT_ROOT / "web"
if WEB_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(WEB_DIR), html=True), name="ui")

# ... (rest of your endpoints remain exactly the same)
MATCHES_CACHE = {"data": [], "timestamp": None, "ttl_minutes": 5}

# Your existing endpoints here (@app.get, @app.post, etc.)
# ... (I'm keeping them the same as your original file)

@app.on_event("startup")
async def startup_event():
    """Runs on API startup"""
    print("=" * 60)
    print("🚀 Predicta API Starting...")
    print("=" * 60)
    print(f"📁 Project Root: {PROJECT_ROOT}")
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
    elif PREDICTOR_TYPE == "standard" and PREDICTOR and hasattr(PREDICTOR, 'best_models_per_class'):
        if PREDICTOR.best_models_per_class:
            print(f"🧠 Smart Selection: ACTIVE")
    
    print("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_weighted:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )

import sys, os, logging, pickle
from pathlib import Path
from datetime import datetime, timedelta

# ---------- PATH fix ----------
BASE_DIR = Path(__file__).resolve().parent
APP_DIR = BASE_DIR / "app"
if str(BASE_DIR) not in sys.path: sys.path.insert(0, str(BASE_DIR))
if str(APP_DIR) not in sys.path: sys.path.insert(0, str(APP_DIR))

# ---------- FastAPI ----------
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
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


# ---------- IMPROVED Model Loading Functions ----------
def safe_pickle_load(filepath, encoding='latin1'):
    """
    Safely load pickle with multiple encoding attempts
    """
    encodings = [encoding, 'bytes', 'ASCII', 'UTF-8']
    
    for enc in encodings:
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f, encoding=enc)
        except (UnicodeDecodeError, KeyError) as e:
            continue
    
    raise ValueError(f"Could not load pickle with any encoding: {encodings}")


def rebuild_model_from_scratch(model_path):
    """
    If model is corrupted, retrain a minimal version
    """
    print("🔄 Model corrupted, creating new lightweight model...")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.calibration import CalibratedClassifierCV
        import pandas as pd
        import numpy as np
        
        # Load dataset
        df = pd.read_csv(MERGED_DATA_FILE)
        
        # Create minimal weighted feature engineer
        from app.weighted_feature_engineer import WeightedFeatureEngineer
        fe = WeightedFeatureEngineer(df=df)
        
        print("📊 Extracting features for training...")
        features_list = []
        labels = []
        
        # Sample data for quick training (use subset for speed)
        sample_size = min(10000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        
        for idx, row in df_sample.iterrows():
            try:
                odds = {
                    "1": row.get('B365H', 2.5),
                    "X": row.get('B365D', 3.2),
                    "2": row.get('B365A', 2.8)
                }
                
                features = fe.extract_match_features(
                    row['HomeTeam'],
                    row['AwayTeam'],
                    odds
                )
                
                if features:
                    features_list.append(features)
                    
                    # Determine label
                    ftr = row.get('FTR', 'D')
                    if ftr == 'H':
                        labels.append(0)  # Home win
                    elif ftr == 'D':
                        labels.append(1)  # Draw
                    else:
                        labels.append(2)  # Away win
                        
            except Exception as e:
                continue
        
        if len(features_list) < 100:
            raise ValueError("Insufficient valid training data")
        
        # Convert to DataFrame
        X = pd.DataFrame(features_list)
        y = np.array(labels)
        
        print(f"✅ Extracted {len(X)} samples with {len(X.columns)} features")
        
        # Train lightweight model
        base_model = RandomForestClassifier(
            n_estimators=50,  # Reduced for speed
            max_depth=10,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Calibrate probabilities
        model = CalibratedClassifierCV(base_model, cv=3, method='isotonic')
        
        print("🎯 Training model...")
        model.fit(X, y)
        
        # Save model
        model_data = {
            'model': model,
            'feature_names': list(X.columns),
            'metadata': {
                'training_date': datetime.now().isoformat(),
                'n_samples': len(X),
                'n_features': len(X.columns),
                'type': 'emergency_rebuild',
                'scikit_learn_version': '1.3.0'
            }
        }
        
        import joblib
        joblib.dump(model_data, model_path, compress=3, protocol=4)
        
        print(f"✅ Emergency model trained and saved: {model_path}")
        return model_data
        
    except Exception as e:
        print(f"❌ Emergency model creation failed: {e}")
        return None


def load_weighted_model_with_advanced_fallback():
    """
    Advanced model loading with multiple fallback strategies
    """
    model_path = BASE_DIR / "models" / "weighted_model.pkl"
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        print("🔄 Attempting to build new model...")
        return rebuild_model_from_scratch(model_path)
    
    print(f"✅ Model file found: {model_path}")
    print(f"📏 Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Strategy 1: Standard joblib load
    try:
        import joblib
        model = joblib.load(model_path)
        print(f"✅ Model loaded with joblib.load()")
        return model
    except KeyError as e:
        print(f"⚠️  joblib.load failed with KeyError: {e}")
    except Exception as e:
        print(f"⚠️  joblib.load failed: {e}")
    
    # Strategy 2: Joblib with mmap
    try:
        import joblib
        model = joblib.load(model_path, mmap_mode='r')
        print(f"✅ Model loaded with mmap_mode='r'")
        return model
    except Exception as e:
        print(f"⚠️  mmap_mode failed: {e}")
    
    # Strategy 3: Standard pickle with different encodings
    for encoding in ['latin1', 'bytes', 'ASCII']:
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f, encoding=encoding)
            print(f"✅ Model loaded with pickle, encoding={encoding}")
            return model
        except Exception as e:
            print(f"⚠️  pickle with {encoding} failed: {e}")
            continue
    
    # Strategy 4: Rebuild from scratch
    print("⚠️  All loading methods failed. Rebuilding model...")
    return rebuild_model_from_scratch(model_path)


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
    try:
        import pandas as pd
        if MERGED_DATA_FILE.exists():
            df = pd.read_csv(MERGED_DATA_FILE)
            print(f"📄 Loaded dataset: {MERGED_DATA_FILE} (rows={len(df)})")
            return df
        print("ℹ️  No dataset found; using odds-only features.")
        return None
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return None

DATAFRAME = _load_dataset_df()
FEATURE_ENGINEER = LiveFeatureEngineer(df=DATAFRAME)

# ---------- Initialize Predictor ----------
print("\n🎯 Initializing Weighted Predictor...")

if WEIGHTED_AVAILABLE:
    try:
        model_dir = BASE_DIR / "models"
        model_path = model_dir / "weighted_model.pkl"
        
        # Use advanced loading function
        loaded_model_data = load_weighted_model_with_advanced_fallback()
        
        if loaded_model_data:
            PREDICTOR = WeightedPredictor(
                model_dir=model_dir,
                draw_threshold=0.30,
                enable_monitoring=True
            )
            
            # Manually set the model
            if isinstance(loaded_model_data, dict):
                PREDICTOR.model = loaded_model_data.get('model')
                PREDICTOR.feature_names = loaded_model_data.get('feature_names', [])
                PREDICTOR.metadata = loaded_model_data.get('metadata', {})
            else:
                PREDICTOR.model = loaded_model_data
            
            PREDICTOR_TYPE = "weighted"
            print("✅ Weighted predictor initialized successfully")
        else:
            raise Exception("Could not load or create model")
            
    except Exception as e:
        print(f"⚠️  Weighted predictor failed: {e}")
        import traceback
        traceback.print_exc()
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
            PREDICTOR_TYPE = "no_model"
    except Exception as e:
        print(f"❌ Standard predictor failed: {e}")
        PREDICTOR_TYPE = "no_model"

# Final fallback - simple predictor
if PREDICTOR is None:
    print("📄 Creating simple odds-based predictor...")
    try:
        class SimplePredictor:
            def predict_match(self, home_team, away_team, odds=None, **kwargs):
                odds = odds or DEFAULT_ODDS
                try:
                    o1, ox, o2 = float(odds["1"]), float(odds["X"]), float(odds["2"])
                    tot = (1/o1 + 1/ox + 1/o2)
                    p1, px, p2 = (1/o1)/tot, (1/ox)/tot, (1/o2)/tot
                    
                    max_prob = max(p1, px, p2)
                    if max_prob == p1:
                        prediction = "1"
                    elif max_prob == px:
                        prediction = "X" 
                    else:
                        prediction = "2"
                    
                    return {
                        "prediction": prediction,
                        "confidence": max_prob,
                        "probabilities": {
                            "home_win": p1,
                            "draw": px, 
                            "away_win": p2
                        },
                        "prediction_method": "odds_only",
                        "warning": "Using simple odds-based prediction (no ML model)"
                    }
                except Exception as e:
                    return {
                        "prediction": "?",
                        "confidence": 0.0,
                        "error": str(e),
                        "prediction_method": "error"
                    }
        
        PREDICTOR = SimplePredictor()
        PREDICTOR_TYPE = "simple_odds"
        print("✅ Simple odds-based predictor initialized")
        
    except Exception as e:
        print(f"❌ Even simple predictor failed: {e}")

# ---------- FastAPI App ----------
app = FastAPI(
    title="Predicta Weighted API",
    description="Football match prediction with weighted feature system (75% Odds, 15% H2H, 10% Form)",
    version="3.2"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# Static files serving
@app.get("/")
def serve_index():
    """Serve the main frontend"""
    index_path = BASE_DIR / "index_weighted.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Predicta API - Frontend not found"}

@app.get("/index_weighted.html")
def serve_index_html():
    """Serve the main frontend"""
    index_path = BASE_DIR / "index_weighted.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"error": "Frontend not found"}

# Static files
WEB_DIR = BASE_DIR
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

MATCHES_CACHE = {"data": [], "timestamp": None, "ttl_minutes": 5}

# ... (rest of the endpoints remain the same as in your original file)
# I'll include the key ones:

@app.get("/api/status")
def api_status():
    """API status and model information"""
    status = {
        "status": "running",
        "predictor_type": PREDICTOR_TYPE,
        "weighted_system": WEIGHTED_AVAILABLE,
        "nesine_fetcher": NESINE_AVAILABLE,
        "web_ui_available": True,
        "dataset_loaded": DATAFRAME is not None,
        "model_loaded": PREDICTOR is not None,
        "environment": "railway"
    }
    
    if PREDICTOR_TYPE == "weighted":
        status["model_quality"] = "premium"
        status["feature_priorities"] = {
            "odds": "75%",
            "h2h": "15%", 
            "form": "10%"
        }
    elif PREDICTOR_TYPE == "simple_odds":
        status["warning"] = "Using simple odds-based prediction"
        status["model_quality"] = "minimal"
    
    return status

# ... (continue with the rest of your endpoints)

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
    print(f"📊 Dataset: {'Loaded' if DATAFRAME is not None else 'Not Loaded'}")
    print("=" * 60)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main_weighted:app", 
        host="0.0.0.0", 
        port=port, 
        reload=False,
        log_level="info"
    )import sys, os, logging, pickle
from pathlib import Path
from datetime import datetime, timedelta

# ---------- PATH fix ----------
BASE_DIR = Path(__file__).resolve().parent
APP_DIR = BASE_DIR / "app"
if str(BASE_DIR) not in sys.path: sys.path.insert(0, str(BASE_DIR))
if str(APP_DIR) not in sys.path: sys.path.insert(0, str(APP_DIR))

# ---------- FastAPI ----------
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
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


# ---------- Model Loading Functions ----------
def load_weighted_model_with_fallback():
    """Model yükleme için gelişmiş fallback sistemi"""
    model_path = BASE_DIR / "models" / "weighted_model.pkl"
    
    if not model_path.exists():
        print(f"❌ Model dosyası bulunamadı: {model_path}")
        return None
    
    print(f"✅ Model dosyası bulundu: {model_path}")
    print(f"📁 Boyut: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Farklı yükleme yöntemlerini dene
    load_methods = [
        # 1. Standart pickle
        lambda: pickle.load(open(model_path, 'rb')),
        
        # 2. Joblib ile
        lambda: __import__('joblib').load(model_path),
        
        # 3. Latin1 encoding ile
        lambda: pickle.load(open(model_path, 'rb'), encoding='latin1'),
        
        # 4. Bytes encoding ile
        lambda: pickle.load(open(model_path, 'rb'), encoding='bytes'),
        
        # 5. Ignore errors ile
        lambda: pickle.load(open(model_path, 'rb'), errors='ignore'),
    ]
    
    for i, method in enumerate(load_methods):
        try:
            model = method()
            print(f"✅ Model {i+1}. yöntemle yüklendi")
            return model
        except Exception as e:
            print(f"❌ Yöntem {i+1} başarısız: {e}")
            continue
    
    print("❌ Tüm model yükleme yöntemleri başarısız")
    return None


def create_simple_fallback_model():
    """Acil durum için basit model oluştur"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        
        print("🔄 Basit fallback model oluşturuluyor...")
        
        # Basit bir RandomForest modeli
        model = RandomForestClassifier(
            n_estimators=10,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Fake training data
        X_dummy = np.random.rand(100, 10)
        y_dummy = np.random.randint(0, 3, 100)
        
        model.fit(X_dummy, y_dummy)
        
        # Modeli kaydet
        model_path = BASE_DIR / "models" / "fallback_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f, protocol=4)
            
        print("✅ Basit fallback model oluşturuldu ve kaydedildi")
        return model
        
    except Exception as e:
        print(f"❌ Basit model oluşturma hatası: {e}")
        return None


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
    try:
        import pandas as pd
        if MERGED_DATA_FILE.exists():
            df = pd.read_csv(MERGED_DATA_FILE)
            print(f"📄 Loaded dataset: {MERGED_DATA_FILE} (rows={len(df)})")
            return df
        print("ℹ️  No dataset found; using odds-only features.")
        return None
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
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
            
            # Gelişmiş model yükleme
            loaded_model = load_weighted_model_with_fallback()
            
            if loaded_model:
                PREDICTOR = WeightedPredictor(
                    model_dir=model_dir,
                    draw_threshold=0.30,
                    enable_monitoring=True
                )
                # Modeli manuel olarak set et
                PREDICTOR.model = loaded_model
                PREDICTOR_TYPE = "weighted"
                print("✅ Weighted predictor initialized successfully")
            else:
                print("❌ Model file exists but cannot be loaded")
                # Fallback model oluştur
                fallback_model = create_simple_fallback_model()
                if fallback_model:
                    PREDICTOR = WeightedPredictor(
                        model_dir=model_dir,
                        draw_threshold=0.30,
                        enable_monitoring=True
                    )
                    PREDICTOR.model = fallback_model
                    PREDICTOR_TYPE = "weighted_fallback"
                    print("✅ Fallback weighted predictor initialized")
                else:
                    raise Exception("Cannot load model or create fallback")
                    
        else:
            print(f"⚠️  Weighted model not found at {model_path}")
            # Fallback model oluştur
            fallback_model = create_simple_fallback_model()
            if fallback_model:
                PREDICTOR = WeightedPredictor(
                    model_dir=model_dir,
                    draw_threshold=0.30,
                    enable_monitoring=True
                )
                PREDICTOR.model = fallback_model
                PREDICTOR_TYPE = "weighted_fallback"
                print("✅ Fallback weighted predictor initialized (no trained model)")
            else:
                raise FileNotFoundError("Weighted model not trained yet")
            
    except Exception as e:
        print(f"⚠️  Weighted predictor failed: {e}")
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
            PREDICTOR_TYPE = "no_model"
    except Exception as e:
        print(f"❌ Standard predictor failed: {e}")
        PREDICTOR_TYPE = "no_model"

# Final fallback - basit predictor
if PREDICTOR is None and PREDICTOR_TYPE == "no_model":
    print("🔄 Creating simple odds-based predictor...")
    try:
        # Basit odds-based predictor
        class SimplePredictor:
            def predict_match(self, home_team, away_team, odds=None, **kwargs):
                odds = odds or DEFAULT_ODDS
                try:
                    o1, ox, o2 = float(odds["1"]), float(odds["X"]), float(odds["2"])
                    tot = (1/o1 + 1/ox + 1/o2)
                    p1, px, p2 = (1/o1)/tot, (1/ox)/tot, (1/o2)/tot
                    
                    # Basit kural: en yüksek probability kazanır
                    max_prob = max(p1, px, p2)
                    if max_prob == p1:
                        prediction = "1"
                    elif max_prob == px:
                        prediction = "X" 
                    else:
                        prediction = "2"
                    
                    return {
                        "prediction": prediction,
                        "confidence": max_prob,
                        "probabilities": {
                            "home_win": p1,
                            "draw": px, 
                            "away_win": p2
                        },
                        "prediction_method": "odds_only",
                        "warning": "Using simple odds-based prediction (no ML model)"
                    }
                except Exception as e:
                    return {
                        "prediction": "?",
                        "confidence": 0.0,
                        "error": str(e),
                        "prediction_method": "error"
                    }
        
        PREDICTOR = SimplePredictor()
        PREDICTOR_TYPE = "simple_odds"
        print("✅ Simple odds-based predictor initialized")
        
    except Exception as e:
        print(f"❌ Even simple predictor failed: {e}")

# ---------- FastAPI App ----------
app = FastAPI(
    title="Predicta Weighted API",
    description="Football match prediction with weighted feature system (75% Odds, 15% H2H, 10% Form)",
    version="3.1"  # Version updated
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# Static files serving - Railway için
@app.get("/")
def serve_index():
    """Serve the main frontend"""
    index_path = BASE_DIR / "index_weighted.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Predicta API - Frontend not found"}

@app.get("/index_weighted.html")
def serve_index_html():
    """Serve the main frontend"""
    index_path = BASE_DIR / "index_weighted.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"error": "Frontend not found"}

# Static files
WEB_DIR = BASE_DIR
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

MATCHES_CACHE = {"data": [], "timestamp": None, "ttl_minutes": 5}


@app.get("/api/status")
def api_status():
    """API status and model information"""
    status = {
        "status": "running",
        "predictor_type": PREDICTOR_TYPE,
        "weighted_system": WEIGHTED_AVAILABLE,
        "nesine_fetcher": NESINE_AVAILABLE,
        "web_ui_available": True,  # Artık her zaman mevcut
        "dataset_loaded": DATAFRAME is not None,
        "model_loaded": PREDICTOR is not None,
        "environment": "railway"
    }
    
    if PREDICTOR_TYPE == "no_model":
        status["warning"] = "No trained model available"
        status["instructions"] = {
            "weighted": "python app/weighted_trainer.py",
            "standard": "python train_real_model.py"
        }
    elif "fallback" in PREDICTOR_TYPE:
        status["warning"] = "Using fallback model"
        status["model_quality"] = "basic"
    elif PREDICTOR_TYPE == "simple_odds":
        status["warning"] = "Using simple odds-based prediction"
        status["model_quality"] = "minimal"
    elif PREDICTOR_TYPE == "weighted":
        weighted_model_path = BASE_DIR / "models" / "weighted_model.pkl"
        status["model_path"] = str(weighted_model_path)
        status["model_trained"] = weighted_model_path.exists()
        status["feature_priorities"] = {
            "odds": "75%",
            "h2h": "15%", 
            "form": "10%"
        }
        status["model_quality"] = "premium"
    else:
        try:
            from app.config import MODEL_PATH
        except ImportError:
            MODEL_PATH = BASE_DIR / "models" / "model.pkl"
        status["model_path"] = str(MODEL_PATH)
        status["model_trained"] = MODEL_PATH.exists()
        if PREDICTOR and hasattr(PREDICTOR, 'best_models_per_class'):
            status["smart_selection_enabled"] = bool(PREDICTOR.best_models_per_class)
        status["model_quality"] = "standard"
    
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
    
    if PREDICTOR is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "No predictor available",
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
        "data_source": resp.get("source", "unknown"),
        "model_quality": "premium" if PREDICTOR_TYPE == "weighted" else "basic"
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
    
    if PREDICTOR is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "No predictor available",
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
        # Basit stats için
        stats = {
            "predictor_type": PREDICTOR_TYPE,
            "model_quality": "premium" if PREDICTOR_TYPE == "weighted" else "basic",
            "predictions_available": True,
            "last_updated": datetime.now().isoformat()
        }
        
        if hasattr(PREDICTOR, 'get_performance_stats'):
            detailed_stats = PREDICTOR.get_performance_stats()
            stats.update(detailed_stats)
        
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


@app.get("/health")
def health_check():
    """Health check endpoint for Railway"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.1",
        "environment": "railway"
    }


@app.on_event("startup")
async def startup_event():
    """Runs on API startup with auto-training"""
    global PREDICTOR, PREDICTOR_TYPE, WEIGHTED_AVAILABLE
    
    print("=" * 60)
    print("🚀 Predicta API Starting...")
    print("=" * 60)
    
    model_path = BASE_DIR / "models" / "weighted_model.pkl"
    
    # Model yoksa veya bozuksa eğit
    if not model_path.exists():
        print("⚠️  No model found, training now...")
        
        if DATAFRAME is not None and len(DATAFRAME) > 0:
            try:
                print("📊 Training model with dataset...")
                print(f"   Dataset size: {len(DATAFRAME)} matches")
                
                # Import and train
                from app.weighted_trainer import train_model
                
                # Train the model
                train_model()
                
                # Reload predictor
                if model_path.exists():
                    print("✅ Model trained successfully, loading...")
                    PREDICTOR = WeightedPredictor(
                        model_dir=BASE_DIR / "models",
                        draw_threshold=0.30,
                        enable_monitoring=True
                    )
                    PREDICTOR_TYPE = "weighted"
                    WEIGHTED_AVAILABLE = True
                    print("✅ Weighted predictor loaded")
                else:
                    print("⚠️  Training completed but model file not found")
                    
            except Exception as e:
                print(f"❌ Training failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("⚠️  No dataset available for training")
    else:
        # Try to load existing model
        try:
            import joblib
            test = joblib.load(model_path)
            print("✅ Existing model is valid")
        except Exception as e:
            print(f"❌ Model corrupted: {e}")
            print("🔄 Deleting and retraining...")
            model_path.unlink()
            
            # Train new model
            if DATAFRAME is not None:
                try:
                    from app.weighted_trainer import train_model
                    train_model()
                    
                    if model_path.exists():
                        PREDICTOR = WeightedPredictor(
                            model_dir=BASE_DIR / "models",
                            draw_threshold=0.30,
                            enable_monitoring=True
                        )
                        PREDICTOR_TYPE = "weighted"
                        WEIGHTED_AVAILABLE = True
                        print("✅ New model trained and loaded")
                except Exception as train_err:
                    print(f"❌ Retraining failed: {train_err}")
    
    print(f"📁 Base Directory: {BASE_DIR}")
    print(f"🎯 Predictor Type: {PREDICTOR_TYPE.upper()}")
    print(f"⚖️  Weighted System: {'ENABLED' if WEIGHTED_AVAILABLE else 'DISABLED'}")
    print(f"🌐 Nesine Fetcher: {'ENABLED ✅' if NESINE_AVAILABLE else 'DISABLED ⚠️'}")
    print(f"🤖 Model Status: {'LOADED ✅' if PREDICTOR else 'NOT LOADED ⚠️'}")
    print(f"📊 Dataset: {'Loaded' if DATAFRAME is not None else 'Not Loaded'}")
    
    if PREDICTOR_TYPE == "weighted":
        print(f"📈 Feature Priorities: 75% Odds | 15% H2H | 10% Form")
    
    print("=" * 60)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main_weighted:app", 
        host="0.0.0.0", 
        port=port, 
        reload=False,  # Railway'de reload kapalı
        log_level="info"
    )

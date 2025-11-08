"""
Weighted Predictor with detailed error logging
"""
import joblib
import traceback
import sys
from pathlib import Path

class WeightedPredictor:
    def __init__(self, model_dir, draw_threshold=0.30, enable_monitoring=True):
        print("🎯 Weighted Predictor Initializing...")
        print(f"   Model dir: {model_dir}")
        print(f"   Draw threshold: {draw_threshold}")
        
        self.model_dir = Path(model_dir)
        self.draw_threshold = draw_threshold
        self.enable_monitoring = enable_monitoring
        self.model = None
        self.feature_names = None
        self.metadata = {}
        
        # Load model with detailed error handling
        model_path = self.model_dir / "weighted_model.pkl"
        
        print(f"✅ Loading model: {model_path}")
        print(f"   File exists: {model_path.exists()}")
        
        if model_path.exists():
            file_size = model_path.stat().st_size / (1024 * 1024)
            print(f"   File size: {file_size:.2f} MB")
            
            # Check available memory
            try:
                import psutil
                mem = psutil.virtual_memory()
                print(f"   Available RAM: {mem.available / (1024**2):.2f} MB")
                print(f"   Total RAM: {mem.total / (1024**2):.2f} MB")
                print(f"   Used RAM: {mem.used / (1024**2):.2f} MB")
            except ImportError:
                print("   (psutil not available, can't check memory)")
        
        try:
            # Try loading with different methods
            print("\n🔄 Attempting to load model...")
            
            # Method 1: Standard joblib load
            try:
                model_data = joblib.load(model_path)
                print("✅ Model loaded successfully with joblib")
            except Exception as e1:
                print(f"❌ joblib.load failed: {e1}")
                
                # Method 2: Try with mmap_mode for large files
                print("🔄 Trying with mmap_mode...")
                try:
                    import numpy as np
                    model_data = joblib.load(model_path, mmap_mode='r')
                    print("✅ Model loaded with mmap_mode")
                except Exception as e2:
                    print(f"❌ mmap_mode failed: {e2}")
                    raise
            
            # Extract model components
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.feature_names = model_data.get('feature_names', [])
                self.metadata = model_data.get('metadata', {})
                print(f"✅ Model components extracted")
                print(f"   Model type: {type(self.model)}")
                print(f"   Features: {len(self.feature_names)}")
                print(f"   Metadata keys: {list(self.metadata.keys())}")
            else:
                self.model = model_data
                print(f"⚠️  Model is not a dict, loaded as: {type(self.model)}")
            
            if self.model is None:
                raise ValueError("Model is None after loading")
            
            print("✅ Weighted Predictor initialized successfully")
            
        except MemoryError as e:
            print(f"❌ MEMORY ERROR: {e}")
            print(f"   Model size ({file_size:.2f} MB) too large for available RAM")
            print(f"   Solution: Use a smaller/optimized model or upgrade instance")
            raise
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Full traceback:")
            traceback.print_exc()
            
            # Additional diagnostics
            print("\n🔍 Diagnostics:")
            print(f"   Python version: {sys.version}")
            print(f"   Joblib version: {joblib.__version__}")
            
            try:
                import sklearn
                print(f"   Scikit-learn version: {sklearn.__version__}")
            except:
                print("   Scikit-learn: not available")
            
            raise
    
    def predict_match(self, home_team, away_team, odds=None, feature_engineer=None):
        """Predict match outcome"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Extract features
        if feature_engineer is None:
            raise ValueError("Feature engineer required")
        
        features = feature_engineer.extract_match_features(
            home_team=home_team,
            away_team=away_team,
            odds=odds
        )
        
        # Convert to array
        import numpy as np
        X = np.array([list(features.values())])
        
        # Predict
        probabilities = self.model.predict_proba(X)[0]
        prediction = self.model.predict(X)[0]
        
        # Map to outcome
        outcome_map = {0: '1', 1: 'X', 2: '2'}
        predicted_outcome = outcome_map.get(prediction, '?')
        
        return {
            'prediction': predicted_outcome,
            'prediction_name': {
                '1': 'Home Win',
                'X': 'Draw', 
                '2': 'Away Win'
            }.get(predicted_outcome, 'Unknown'),
            'confidence': float(max(probabilities)),
            'probabilities': {
                'home_win': float(probabilities[0]),
                'draw': float(probabilities[1]),
                'away_win': float(probabilities[2])
            },
            'odds_confidence': odds.get('market_confidence', 0.5) if isinstance(odds, dict) else 0.5,
            'feature_priorities': {
                'odds': '75%',
                'h2h': '15%',
                'form': '10%'
            }
        }
    
    def get_performance_stats(self):
        """Get performance statistics"""
        return {
            'model_loaded': self.model is not None,
            'metadata': self.metadata,
            'n_features': len(self.feature_names) if self.feature_names else 0
        }

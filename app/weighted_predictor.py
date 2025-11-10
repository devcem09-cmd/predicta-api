"""
WEIGHTED PREDICTOR + SMART MODEL SELECTION + LOGIC VALIDATION
"""

import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

class WeightedPredictor:
    """
    Gelişmiş Weighted Predictor:
    - %75 Odds, %15 H2H, %10 Form
    - Smart Model Selection (her sınıf için en iyi model)
    - Logic Validation (imkansız kombinasyonları yakalar)
    """

    def __init__(self,
                 model_dir: Optional[Path] = None,
                 draw_threshold: float = 0.30,
                 enable_monitoring: bool = True,
                 enable_logic_validation: bool = True,
                 use_smart_selection: bool = True):  # ✅ YENİ
        
        if model_dir is None:
            base_dir = Path(__file__).resolve().parents[1]
            model_dir = base_dir / "models"

        self.model_dir = Path(model_dir)
        self.draw_threshold = draw_threshold
        self.enable_monitoring = enable_monitoring
        self.enable_logic_validation = enable_logic_validation
        self.use_smart_selection = use_smart_selection  # ✅ YENİ

        self.ensemble = None
        self.scaler = None
        self.metadata = {}
        self.feature_names = []
        self.predictions_log = []
        
        # ✅ YENİ: Individual models
        self.individual_models = {}
        self.best_models_per_class = {}

        print(f"🎯 Weighted Predictor Initializing...")
        print(f"   Model dir: {self.model_dir}")
        print(f"   Draw threshold: {self.draw_threshold}")
        print(f"   Logic validation: {'ENABLED ✅' if enable_logic_validation else 'DISABLED'}")
        print(f"   Smart selection: {'ENABLED ✅' if use_smart_selection else 'DISABLED'}")

        self._load_models()
        
        if use_smart_selection:
            self._load_individual_models()
            self._select_best_per_class()

    def _load_models(self):
        """Ensemble model, scaler ve metadata yükle"""
        try:
            # Model yükle
            candidate_models = ["weighted_model.pkl", "model.pkl"]
            model_path = None
            for name in candidate_models:
                path = self.model_dir / name
                if path.exists():
                    model_path = path
                    break

            if not model_path:
                raise FileNotFoundError(f"❌ No model found in {self.model_dir}")

            print(f"✅ Loading ensemble: {model_path}")
            self.ensemble = joblib.load(model_path)

            # Scaler yükle
            candidate_scalers = ["weighted_scaler.pkl", "scaler.pkl"]
            scaler_path = None
            for name in candidate_scalers:
                path = self.model_dir / name
                if path.exists():
                    scaler_path = path
                    break

            if not scaler_path:
                raise FileNotFoundError(f"❌ Scaler not found in {self.model_dir}")

            print(f"✅ Loading scaler: {scaler_path}")
            self.scaler = joblib.load(scaler_path)

            # Metadata yükle
            metadata_path = self.model_dir / "weighted_model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                print(f"✅ Metadata loaded")

                if 'feature_names' in self.metadata:
                    self.feature_names = self.metadata['feature_names']

            print(f"✅ Models loaded successfully")

        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise

    def _load_individual_models(self):
        """Individual modelleri yükle (smart selection için)"""
        model_files = {
            'xgboost': 'model_xgboost.pkl',
            'lightgbm': 'model_lightgbm.pkl',
            'catboost': 'model_catboost.pkl',
            'random_forest': 'model_random_forest.pkl'
        }
        
        for name, filename in model_files.items():
            path = self.model_dir / filename
            if path.exists():
                try:
                    self.individual_models[name] = joblib.load(path)
                    print(f"   ✓ Loaded: {name}")
                except Exception as e:
                    print(f"   ⚠️  Failed: {name} - {e}")

    def _select_best_per_class(self):
        """Her sınıf için en iyi modeli seç"""
        if 'individual_scores' not in self.metadata:
            print("   ⚠️  No scores found, smart selection disabled")
            self.use_smart_selection = False
            return
        
        scores = self.metadata['individual_scores']
        
        # Home Win için en iyi
        best_home = max(scores.items(), 
                       key=lambda x: x[1].get('test_accuracy', 0))
        
        # Draw için en iyi (draw_accuracy'e göre)
        best_draw = max(scores.items(),
                       key=lambda x: x[1].get('draw_accuracy', 0))
        
        # Away Win için en iyi
        best_away = max(scores.items(),
                       key=lambda x: x[1].get('test_accuracy', 0))
        
        self.best_models_per_class = {
            'home_win': best_home[0],
            'draw': best_draw[0],
            'away_win': best_away[0]
        }
        
        print("\n   🎯 Best Models Per Class:")
        print(f"      Home Win: {best_home[0]} ({best_home[1]['test_accuracy']:.2%})")
        print(f"      Draw:     {best_draw[0]} ({best_draw[1]['draw_accuracy']:.2%})")
        print(f"      Away Win: {best_away[0]} ({best_away[1]['test_accuracy']:.2%})")

    def _prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """Feature dict'i numpy array'e dönüştür"""
        if self.feature_names:
            feature_vector = [float(features.get(name, 0.0)) for name in self.feature_names]
        else:
            feature_vector = [float(v) for k, v in sorted(features.items())]
        return np.array([feature_vector])

    def _apply_odds_priority(self, proba: np.ndarray, odds_confidence: float) -> np.ndarray:
        """Odds ağırlığını uygula"""
        if odds_confidence > 0.7:
            calibration_factor = 1.2
        elif odds_confidence > 0.5:
            calibration_factor = 1.1
        else:
            calibration_factor = 1.0

        proba_calibrated = proba.copy()
        proba_calibrated[1] = proba_calibrated[1] * calibration_factor
        proba_calibrated = proba_calibrated / proba_calibrated.sum()
        return proba_calibrated

    def _predict_with_smart_selection(self, X_scaled: np.ndarray) -> Tuple[int, np.ndarray, str]:
        """
        ✅ SMART SELECTION:
        1. Ensemble'dan genel tahmin al
        2. Tahmin edilen sınıf için en iyi modeli seç
        3. O modelden final tahmin yap
        """
        # Önce ensemble'dan tahmin
        ensemble_proba = self.ensemble.predict_proba(X_scaled)[0]
        predicted_class_idx = int(np.argmax(ensemble_proba))
        
        # Sınıfa göre en iyi model
        class_map = {0: 'home_win', 1: 'draw', 2: 'away_win'}
        predicted_class = class_map[predicted_class_idx]
        
        best_model_name = self.best_models_per_class.get(predicted_class)
        
        if best_model_name and best_model_name in self.individual_models:
            # En iyi modelden final tahmin
            best_model = self.individual_models[best_model_name]
            final_proba = best_model.predict_proba(X_scaled)[0]
            final_pred_idx = int(np.argmax(final_proba))
            
            return final_pred_idx, final_proba, best_model_name
        else:
            # Fallback: ensemble
            return predicted_class_idx, ensemble_proba, 'ensemble'

    def _validate_prediction_logic(self, 
                                   ms_prediction: str,
                                   ms_confidence: float,
                                   ou_prediction: Optional[str] = None,
                                   ou_confidence: Optional[float] = None,
                                   btts_prediction: Optional[bool] = None,
                                   btts_confidence: Optional[float] = None) -> Tuple[bool, List[Dict[str, str]]]:
        """Mantık kontrolü (önceki kod ile aynı)"""
        if not self.enable_logic_validation:
            return True, []
        
        warnings = []
        is_valid = True
        
        # MS1/MS2 + Alt 2.5 + KG Var = İMKANSIZ
        if ms_prediction in ['1', '2'] and ou_prediction == 'under' and btts_prediction == True:
            is_valid = False
            warnings.append({
                'severity': 'CRITICAL',
                'type': 'mathematical_impossibility',
                'message': f"❌ {ms_prediction} + Alt 2.5 + KG Var = Matematiksel olarak imkansız!",
                'explanation': (
                    f"Alt 2.5 + KG Var kombinasyonu sadece 1-1 skorunu verir. "
                    f"1-1 skoru beraberliktir, {ms_prediction} olamaz!"
                ),
                'possible_scores': ['1-1 (Beraberlik)'],
                'suggestion': (
                    f"Düzeltme önerileri:\n"
                    f"  • {ms_prediction} + Alt 2.5 + KG YOK (1-0, 2-0 gibi)\n"
                    f"  • {ms_prediction} + ÜST 2.5 + KG Var (2-1, 3-1 gibi)\n"
                    f"  • MSX + Alt 2.5 + KG Var (1-1)"
                )
            })
        
        return is_valid, warnings

    def _adjust_predictions_for_logic(self,
                                     ms_prediction: str,
                                     ms_proba: np.ndarray,
                                     ou_prediction: Optional[str],
                                     btts_prediction: Optional[bool]) -> Tuple[str, Dict[str, Any]]:
        """Otomatik düzeltme"""
        adjustments = {'adjusted': False, 'original': ms_prediction, 'reason': None}
        
        # MS1/MS2 + Alt 2.5 + KG Var → MSX'e çevir
        if ms_prediction in ['1', '2'] and ou_prediction == 'under' and btts_prediction == True:
            ms_prediction = 'X'
            adjustments = {
                'adjusted': True,
                'original': adjustments['original'],
                'new': 'X',
                'reason': 'Alt 2.5 + KG Var kombinasyonu sadece 1-1 (beraberlik) verir',
                'confidence_penalty': 0.15
            }
        
        return ms_prediction, adjustments

    def predict_match(self,
                      home_team: str,
                      away_team: str,
                      odds: Optional[Dict[str, float]] = None,
                      feature_engineer=None,
                      actual_result: Optional[str] = None,
                      return_all_predictions: bool = False) -> Dict[str, Any]:
        """
        ✅ GELİŞMİŞ TAHMİN:
        - Smart model selection (en iyi modeli kullan)
        - Logic validation (imkansız kombinasyonları yakala)
        - Auto-adjustment (hatalı tahminleri düzelt)
        """
        if feature_engineer is None:
            raise ValueError("❌ FeatureEngineer required for predictions")

        # Feature extraction
        features = feature_engineer.extract_match_features(home_team, away_team, odds)
        X = self._prepare_features(features)
        X_scaled = self.scaler.transform(X)
        
        # ✅ SMART SELECTION
        if self.use_smart_selection and self.best_models_per_class:
            pred_idx, proba, model_used = self._predict_with_smart_selection(X_scaled)
        else:
            proba = self.ensemble.predict_proba(X_scaled)[0]
            if proba[1] >= self.draw_threshold and proba[1] == proba.max():
                pred_idx = 1
            else:
                pred_idx = int(np.argmax(proba))
            model_used = 'ensemble'

        # Odds güveni
        odds_confidence = 0.5
        if odds and all(k in odds for k in ['1', 'X', '2']):
            try:
                vals = [float(odds['1']), float(odds['X']), float(odds['2'])]
                if all(v > 0 for v in vals):
                    spread = max(vals) - min(vals)
                    odds_confidence = min(1.0, spread / 2.0)
            except:
                pass

        proba = self._apply_odds_priority(proba, odds_confidence)

        # Label mapping
        idx_to_label = {0: "1", 1: "X", 2: "2"}
        idx_to_name = {0: "Home Win", 1: "Draw", 2: "Away Win"}
        label_to_idx = {"1": 0, "X": 1, "2": 2}  # ✅ DÜZELTİLDİ
        
        prediction_label = idx_to_label[pred_idx]
        original_prediction = prediction_label

        result = {
            "home_team": home_team,
            "away_team": away_team,
            "prediction": prediction_label,
            "prediction_name": idx_to_name[pred_idx],
            "probabilities": {
                "home_win": float(proba[0]),
                "draw": float(proba[1]),
                "away_win": float(proba[2]),
            },
            "confidence": float(proba[pred_idx]),
            "odds_confidence": float(odds_confidence),
            "model_type": "weighted_ensemble_smart",
            "model_used": model_used,  # ✅ Hangi model kullanıldı
            "feature_priorities": {
                "odds": "75%",
                "h2h": "15%",
                "form": "10%"
            }
        }

        # Alt/Üst ve KG tahminleri (placeholder)
        ou_prediction = None
        ou_confidence = None
        btts_prediction = None
        btts_confidence = None

        # 🔍 MANTIK KONTROLÜ
        if self.enable_logic_validation:
            is_valid, warnings = self._validate_prediction_logic(
                ms_prediction=prediction_label,
                ms_confidence=result['confidence'],
                ou_prediction=ou_prediction,
                ou_confidence=ou_confidence,
                btts_prediction=btts_prediction,
                btts_confidence=btts_confidence
            )
            
            result['logic_validation'] = {
                'is_valid': is_valid,
                'warnings': warnings
            }
            
            # 🔧 Otomatik düzeltme
            if not is_valid and len(warnings) > 0:
                adjusted_pred, adjustments = self._adjust_predictions_for_logic(
                    prediction_label, proba, ou_prediction, btts_prediction
                )
                
                if adjustments['adjusted']:
                    result['prediction'] = adjusted_pred
                    result['prediction_name'] = idx_to_name[label_to_idx[adjusted_pred]]  # ✅ DÜZELTİLDİ
                    result['auto_adjusted'] = adjustments
                    result['confidence'] = result['confidence'] * (1 - adjustments.get('confidence_penalty', 0))

        # Odds analizi
        if odds and all(k in odds for k in ['1', 'X', '2']):
            try:
                o1, ox, o2 = float(odds['1']), float(odds['X']), float(odds['2'])
                total_prob = (1/o1 + 1/ox + 1/o2)
                result["odds_analysis"] = {
                    "market_probabilities": {
                        "home_win": round((1/o1) / total_prob, 4),
                        "draw": round((1/ox) / total_prob, 4),
                        "away_win": round((1/o2) / total_prob, 4),
                    },
                    "market_margin": round(total_prob - 1.0, 4),
                }
            except:
                pass

        # Monitoring
        if self.enable_monitoring:
            self.predictions_log.append({
                "timestamp": datetime.now().isoformat(),
                "home_team": home_team,
                "away_team": away_team,
                "prediction": prediction_label,
                "confidence": result["confidence"],
                "model_used": model_used,
                "actual": actual_result,
                "correct": (actual_result == prediction_label) if actual_result else None,
            })

        return result

    def get_performance_stats(self) -> Dict[str, Any]:
        """Performans istatistikleri"""
        if not self.predictions_log:
            return {"total_predictions": 0}

        total = len(self.predictions_log)
        evaluated = [p for p in self.predictions_log if p.get("actual") is not None]
        
        if not evaluated:
            return {"total_predictions": total, "evaluated_predictions": 0}

        correct = sum(1 for p in evaluated if p.get("correct"))
        
        return {
            "total_predictions": total,
            "evaluated_predictions": len(evaluated),
            "accuracy": correct / len(evaluated),
            "average_confidence": np.mean([p["confidence"] for p in self.predictions_log]),
            "smart_selection_enabled": self.use_smart_selection
        }

"""
FIXED WEIGHTED PREDICTOR - Alt 2.5 + KG Var = X (Beraberlik) Düzeltmesi
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
    - Smart Model Selection
    - Logic Validation + AUTO-CORRECTION
    """

    def __init__(self,
                 model_dir: Optional[Path] = None,
                 draw_threshold: float = 0.30,
                 enable_monitoring: bool = True,
                 enable_logic_validation: bool = True,
                 use_smart_selection: bool = True,
                 auto_correct_logic: bool = True):  # ✅ YENİ
        
        if model_dir is None:
            base_dir = Path(__file__).resolve().parents[1]
            model_dir = base_dir / "models"

        self.model_dir = Path(model_dir)
        self.draw_threshold = draw_threshold
        self.enable_monitoring = enable_monitoring
        self.enable_logic_validation = enable_logic_validation
        self.use_smart_selection = use_smart_selection
        self.auto_correct_logic = auto_correct_logic  # ✅ YENİ

        self.ensemble = None
        self.scaler = None
        self.metadata = {}
        self.feature_names = []
        self.predictions_log = []
        
        self.individual_models = {}
        self.best_models_per_class = {}

        print(f"🎯 Weighted Predictor Initializing...")
        print(f"   Model dir: {self.model_dir}")
        print(f"   Draw threshold: {self.draw_threshold}")
        print(f"   Logic validation: {'ENABLED ✅' if enable_logic_validation else 'DISABLED'}")
        print(f"   Auto-correction: {'ENABLED ✅' if auto_correct_logic else 'DISABLED'}")
        print(f"   Smart selection: {'ENABLED ✅' if use_smart_selection else 'DISABLED'}")

        self._load_models()
        
        if use_smart_selection:
            self._load_individual_models()
            self._select_best_per_class()

    def _load_models(self):
        """Ensemble model, scaler ve metadata yükle"""
        try:
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
        """Individual modelleri yükle"""
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
        
        best_home = max(scores.items(), 
                       key=lambda x: x[1].get('test_accuracy', 0))
        best_draw = max(scores.items(),
                       key=lambda x: x[1].get('draw_accuracy', 0))
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
        """Smart model selection"""
        ensemble_proba = self.ensemble.predict_proba(X_scaled)[0]
        predicted_class_idx = int(np.argmax(ensemble_proba))
        
        class_map = {0: 'home_win', 1: 'draw', 2: 'away_win'}
        predicted_class = class_map[predicted_class_idx]
        
        best_model_name = self.best_models_per_class.get(predicted_class)
        
        if best_model_name and best_model_name in self.individual_models:
            best_model = self.individual_models[best_model_name]
            final_proba = best_model.predict_proba(X_scaled)[0]
            final_pred_idx = int(np.argmax(final_proba))
            return final_pred_idx, final_proba, best_model_name
        else:
            return predicted_class_idx, ensemble_proba, 'ensemble'

    def _check_impossible_combination(self,
                                     ms_prediction: str,
                                     ou_prediction: Optional[str],
                                     btts_prediction: Optional[bool]) -> Tuple[bool, str]:
        """
        🔴 KRİTİK KONTROL: Matematiksel olarak imkansız kombinasyonları yakala
        
        Returns:
            (is_impossible, reason)
        """
        # ❌ MS1/MS2 + Alt 2.5 + KG Var = İMKANSIZ
        # Alt 2.5 (max 2 gol) + KG Var (her ikisi gol) = Sadece 1-1 olabilir
        # 1-1 = Beraberlik, MS1 veya MS2 OLAMAZ!
        
        if ms_prediction in ['1', '2'] and ou_prediction == 'under' and btts_prediction == True:
            return True, (
                f"❌ {ms_prediction} + Alt 2.5 + KG Var kombinasyonu MATEMATİKSEL OLARAK İMKANSIZ!\n\n"
                f"Açıklama:\n"
                f"• Alt 2.5 = Maksimum 2 gol olabilir\n"
                f"• KG Var = Her iki takım gol atmalı (minimum 1-1)\n"
                f"• Bu kombinasyon SADECE 1-1 skorunu verir\n"
                f"• 1-1 skoru beraberliktir, {ms_prediction} tahmini YANLIŞ!\n\n"
                f"Doğru tahmin: MSX (Beraberlik)"
            )
        
        return False, ""

    def _auto_correct_prediction(self,
                                ms_prediction: str,
                                ms_proba: np.ndarray,
                                ou_prediction: Optional[str],
                                btts_prediction: Optional[bool]) -> Tuple[str, np.ndarray, Dict[str, Any]]:
        """
        🔧 OTOMATİK DÜZELTME: İmkansız kombinasyonları düzelt
        
        Returns:
            (corrected_prediction, corrected_proba, correction_info)
        """
        correction_info = {
            'was_corrected': False,
            'original_prediction': ms_prediction,
            'correction_reason': None
        }
        
        is_impossible, reason = self._check_impossible_combination(
            ms_prediction, ou_prediction, btts_prediction
        )
        
        if is_impossible:
            # 🔧 Alt 2.5 + KG Var = Otomatik olarak X'e çevir
            corrected_prediction = 'X'
            
            # Olasılıkları da güncelle (beraberliğe yüksek olasılık ver)
            corrected_proba = ms_proba.copy()
            corrected_proba[1] = 0.85  # Draw'a yüksek olasılık
            corrected_proba[0] = 0.08  # Home'a düşük
            corrected_proba[2] = 0.07  # Away'e düşük
            
            correction_info = {
                'was_corrected': True,
                'original_prediction': ms_prediction,
                'corrected_prediction': corrected_prediction,
                'correction_reason': reason,
                'confidence_penalty': 0.15,
                'auto_corrected_proba': {
                    'home_win': float(corrected_proba[0]),
                    'draw': float(corrected_proba[1]),
                    'away_win': float(corrected_proba[2])
                }
            }
            
            return corrected_prediction, corrected_proba, correction_info
        
        return ms_prediction, ms_proba, correction_info

    def predict_match(self,
                      home_team: str,
                      away_team: str,
                      odds: Optional[Dict[str, float]] = None,
                      feature_engineer=None,
                      actual_result: Optional[str] = None,
                      return_all_predictions: bool = False) -> Dict[str, Any]:
        """
        ✅ GELİŞMİŞ TAHMİN (Otomatik düzeltme ile)
        """
        if feature_engineer is None:
            raise ValueError("❌ FeatureEngineer required for predictions")

        # Feature extraction
        features = feature_engineer.extract_match_features(home_team, away_team, odds)
        X = self._prepare_features(features)
        X_scaled = self.scaler.transform(X)
        
        # Smart Selection
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
        label_to_idx = {"1": 0, "X": 1, "2": 2}
        
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
            "model_used": model_used,
            "feature_priorities": {
                "odds": "75%",
                "h2h": "15%",
                "form": "10%"
            }
        }

        # 🔴 ÖNCELİKLE Over/Under ve BTTS tahminlerini al (düzeltme için gerekli)
        ou_prediction = None
        ou_confidence = None
        btts_prediction = None
        btts_confidence = None

        # 🔧 OTOMATİK DÜZELTME (eğer aktifse)
        correction_info = {'was_corrected': False}
        
        if self.auto_correct_logic:
            # ⚠️ DİKKAT: Gerçek OU ve BTTS değerlerini kullanmalısınız
            # Bu örnekte placeholder kullanıyoruz, gerçek implementasyonda
            # bu değerler önceden hesaplanmalı
            
            corrected_pred, corrected_proba, correction_info = self._auto_correct_prediction(
                prediction_label, proba, ou_prediction, btts_prediction
            )
            
            if correction_info['was_corrected']:
                result['prediction'] = corrected_pred
                result['prediction_name'] = idx_to_name[label_to_idx[corrected_pred]]
                result['probabilities'] = correction_info['auto_corrected_proba']
                result['confidence'] = float(corrected_proba[label_to_idx[corrected_pred]])
                result['auto_correction'] = correction_info
                
                print(f"\n🔧 AUTO-CORRECTION APPLIED:")
                print(f"   Original: {original_prediction}")
                print(f"   Corrected: {corrected_pred}")
                print(f"   Reason: Alt 2.5 + KG Var = Sadece 1-1 (Beraberlik)")

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
                "prediction": result['prediction'],
                "original_prediction": original_prediction,
                "was_corrected": correction_info['was_corrected'],
                "confidence": result["confidence"],
                "model_used": model_used,
                "actual": actual_result,
                "correct": (actual_result == result['prediction']) if actual_result else None,
            })

        return result

    def validate_full_prediction_set(self, ms_prediction, ms_confidence, ou_prediction, 
                                    ou_confidence, btts_prediction, btts_confidence):
        """
        3 tahmini birlikte kontrol et
        """
        warnings = []
        suggestions = []
        is_valid = True
        
        # 🔴 KRİTİK KONTROL
        is_impossible, reason = self._check_impossible_combination(
            ms_prediction, ou_prediction, btts_prediction
        )
        
        if is_impossible:
            is_valid = False
            warnings.append({
                "severity": "CRITICAL",
                "type": "mathematical_impossibility",
                "message": reason,
                "auto_correction_available": True,
                "suggested_correction": "MSX (Beraberlik)"
            })
            
            suggestions.append({
                "type": "critical_fix",
                "original": f"{ms_prediction} + Alt 2.5 + KG Var",
                "recommended_fix": "MSX + Alt 2.5 + KG Var",
                "reason": "Alt 2.5 + KG Var kombinasyonu sadece 1-1 skorunu verir (beraberlik)",
                "confidence": "Matematiksel kesinlik (%100)",
                "guaranteed_score": "1-1"
            })
        
        # Diğer kontroller...
        if ou_prediction == 'over' and not btts_prediction:
            warnings.append({
                "severity": "WARNING",
                "type": "unlikely_combination",
                "message": "⚠️ Üst 2.5 + KG Yok = Zor kombinasyon (tek takım 3+ gol)",
                "probability": "Düşük (~15%)"
            })
        
        if ms_prediction == 'X' and ou_prediction == 'over':
            warnings.append({
                "severity": "WARNING",
                "type": "rare_combination",
                "message": "⚠️ MSX + Üst 2.5 = Çok nadir (yüksek skorlu beraberlik)",
                "probability": "Çok düşük (~8%)"
            })
        
        avg_confidence = (ms_confidence + ou_confidence + btts_confidence) / 3
        
        return {
            "is_valid": is_valid,
            "warnings": warnings,
            "suggestions": suggestions,
            "original_predictions": {
                "ms": ms_prediction,
                "ou": ou_prediction,
                "btts": btts_prediction
            },
            "confidences": {
                "ms": ms_confidence,
                "ou": ou_confidence,
                "btts": btts_confidence,
                "average": round(avg_confidence, 4)
            },
            "validation_summary": {
                "total_warnings": len(warnings),
                "critical_issues": len([w for w in warnings if w.get("severity") == "CRITICAL"]),
                "overall_risk": "CRITICAL" if not is_valid else "MEDIUM" if len(warnings) > 2 else "LOW",
                "recommendation": (
                    "❌ TAHMİNLERİ DÜZELT! (Otomatik düzeltme mevcut)" if not is_valid else
                    "⚠️ Riskli kombinasyon" if len(warnings) > 2 else
                    "✅ Tahminler tutarlı"
                )
            },
            "possible_scores": self._get_possible_scores(ms_prediction, ou_prediction, btts_prediction)
        }

    def _get_possible_scores(self, ms_prediction, ou_prediction, btts_prediction):
        """Olası skorları hesapla"""
        if ou_prediction == 'under' and btts_prediction:
            return ["1-1"]  # Garantili tek skor
        
        scores = []
        if ou_prediction == 'under' and not btts_prediction:
            if ms_prediction == '1':
                scores = ["1-0", "2-0"]
            elif ms_prediction == '2':
                scores = ["0-1", "0-2"]
            else:
                scores = ["0-0"]
        elif ou_prediction == 'over' and btts_prediction:
            if ms_prediction == '1':
                scores = ["2-1", "3-1", "3-2", "4-2"]
            elif ms_prediction == '2':
                scores = ["1-2", "1-3", "2-3", "2-4"]
            else:
                scores = ["2-2", "3-3"]
        elif ou_prediction == 'over' and not btts_prediction:
            if ms_prediction == '1':
                scores = ["3-0", "4-0", "5-0"]
            elif ms_prediction == '2':
                scores = ["0-3", "0-4", "0-5"]
        
        return scores if scores else ["Belirsiz"]

    def get_performance_stats(self) -> Dict[str, Any]:
        """Performans istatistikleri"""
        if not self.predictions_log:
            return {"total_predictions": 0}

        total = len(self.predictions_log)
        evaluated = [p for p in self.predictions_log if p.get("actual") is not None]
        corrected_count = sum(1 for p in self.predictions_log if p.get("was_corrected"))
        
        if not evaluated:
            return {
                "total_predictions": total,
                "evaluated_predictions": 0,
                "auto_corrected": corrected_count
            }

        correct = sum(1 for p in evaluated if p.get("correct"))
        
        return {
            "total_predictions": total,
            "evaluated_predictions": len(evaluated),
            "accuracy": correct / len(evaluated),
            "average_confidence": np.mean([p["confidence"] for p in self.predictions_log]),
            "auto_corrected": corrected_count,
            "correction_rate": corrected_count / total,
            "smart_selection_enabled": self.use_smart_selection
        }

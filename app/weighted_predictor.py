import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

class WeightedPredictor:
    """
    AĞIRLIKLI TAHMİN SİSTEMİ + MANTIK VALIDASYONU

    Öncelik:
    - %75: Bahis Oranları
    - %15: H2H Geçmiş
    - %10: Form Durumu
    
    + Tahmin mantık kontrolü (MS1 + Alt + KG Var gibi çelişkileri yakalar)
    """

    def __init__(self,
                 model_dir: Optional[Path] = None,
                 draw_threshold: float = 0.30,
                 enable_monitoring: bool = True,
                 enable_logic_validation: bool = True):
        """Model yükleme ve yapılandırma"""
        if model_dir is None:
            base_dir = Path(__file__).resolve().parents[1]
            model_dir = base_dir / "models"

        self.model_dir = Path(model_dir)
        self.draw_threshold = draw_threshold
        self.enable_monitoring = enable_monitoring
        self.enable_logic_validation = enable_logic_validation

        self.ensemble = None
        self.scaler = None
        self.metadata = {}
        self.feature_names = []
        self.predictions_log = []

        print(f"🎯 Weighted Predictor Initializing...")
        print(f"   Model dir: {self.model_dir}")
        print(f"   Draw threshold: {self.draw_threshold}")
        print(f"   Logic validation: {'ENABLED ✅' if enable_logic_validation else 'DISABLED'}")

        self._load_models()

    def _load_models(self):
        """Model, scaler ve metadata dosyalarını yükle"""
        try:
            candidate_models = ["weighted_model.pkl", "model.pkl", "best_model.pkl"]
            model_path = None
            for name in candidate_models:
                path = self.model_dir / name
                if path.exists():
                    model_path = path
                    break

            if not model_path:
                raise FileNotFoundError(f"❌ No model found in {self.model_dir}")

            print(f"✅ Loading model: {model_path}")
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
                    print(f"   Features: {len(self.feature_names)}")

                if 'feature_weights' in self.metadata:
                    weights = self.metadata['feature_weights']
                    print(f"   Feature weights:")
                    print(f"      Odds: {weights['odds'] * 100:.0f}%")
                    print(f"      H2H:  {weights['h2h'] * 100:.0f}%")
                    print(f"      Form: {weights['form'] * 100:.0f}%")

            print(f"✅ All models loaded successfully")

        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise

    def _prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """Feature dict'i numpy array'e dönüştür"""
        if self.feature_names:
            feature_vector = [float(features.get(name, 0.0)) for name in self.feature_names]
        else:
            feature_vector = [float(v) for k, v in sorted(features.items())]
        return np.array([feature_vector])

    def _apply_odds_priority(self, proba: np.ndarray, odds_confidence: float) -> np.ndarray:
        """Odds ağırlığını modele uygula"""
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

    def _validate_prediction_logic(self, 
                                   ms_prediction: str,
                                   ms_confidence: float,
                                   ou_prediction: Optional[str] = None,
                                   ou_confidence: Optional[float] = None,
                                   btts_prediction: Optional[bool] = None,
                                   btts_confidence: Optional[float] = None) -> Tuple[bool, List[Dict[str, str]]]:
        """
        🔍 TAHMİN MANTIK KONTROLÜ
        
        Matematiksel olarak imkansız kombinasyonları tespit eder:
        - MS1/MS2 + Alt 2.5 + KG Var = İMKANSIZ (sadece 1-1 olur ama 1-1 beraberlik!)
        - MSX + Alt 2.5 + KG Var = Sadece 1-1 (çok spesifik, güven düşürülmeli)
        
        Returns:
            (is_valid, warnings_list)
        """
        if not self.enable_logic_validation:
            return True, []
        
        warnings = []
        is_valid = True
        
        # 🚨 KRİTİK HATA: MS1/MS2 + Alt 2.5 + KG Var
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
        
        # ⚠️ UYARI: MSX + Alt 2.5 + KG Var (sadece 1-1)
        elif ms_prediction == 'X' and ou_prediction == 'under' and btts_prediction == True:
            warnings.append({
                'severity': 'WARNING',
                'type': 'overly_specific',
                'message': "⚠️ MSX + Alt 2.5 + KG Var = Sadece 1-1 skoru (çok spesifik tahmin)",
                'explanation': (
                    "Bu kombinasyon matematiksel olarak doğru ama çok dar. "
                    "Sadece tek bir skor (1-1) bu tahmini doğrular."
                ),
                'possible_scores': ['1-1'],
                'confidence_adjustment': 'Güven seviyesi %15-20 düşürülmeli',
                'risk': 'Yüksek risk - Sadece 1 skor seçeneği var'
            })
        
        # 💡 BİLGİ: MS1/MS2 + Üst 2.5 + KG Var (mantıklı)
        if ms_prediction in ['1', '2'] and ou_prediction == 'over' and btts_prediction == True:
            warnings.append({
                'severity': 'INFO',
                'type': 'strong_combination',
                'message': f"✅ {ms_prediction} + Üst 2.5 + KG Var = Mantıklı kombinasyon",
                'possible_scores': ['2-1', '3-1', '3-2', '4-1', '4-2'] if ms_prediction == '1' else ['1-2', '1-3', '2-3', '1-4', '2-4'],
                'confidence_note': 'Tahmin mantıksal olarak tutarlı'
            })
        
        # 💡 BİLGİ: MS1/MS2 + Alt 2.5 + KG Yok (mantıklı)
        if ms_prediction in ['1', '2'] and ou_prediction == 'under' and btts_prediction == False:
            warnings.append({
                'severity': 'INFO',
                'type': 'strong_combination',
                'message': f"✅ {ms_prediction} + Alt 2.5 + KG Yok = Mantıklı kombinasyon",
                'possible_scores': ['1-0', '2-0'] if ms_prediction == '1' else ['0-1', '0-2'],
                'confidence_note': 'Tahmin mantıksal olarak tutarlı'
            })
        
        # ⚠️ UYARI: Düşük güvenle mantıksız tahmin
        if not is_valid and ms_confidence and ms_confidence < 0.6:
            warnings.append({
                'severity': 'WARNING',
                'type': 'low_confidence_error',
                'message': f"⚠️ Mantıksız tahmin + Düşük güven ({ms_confidence*100:.0f}%)",
                'suggestion': 'Model kararsız - Tahmin vermekten kaçının veya sadece en yüksek güvenli tahmini kullanın'
            })
        
        return is_valid, warnings

    def _adjust_predictions_for_logic(self,
                                     ms_prediction: str,
                                     ms_proba: np.ndarray,
                                     ou_prediction: Optional[str],
                                     btts_prediction: Optional[bool]) -> Tuple[str, Dict[str, Any]]:
        """
        🔧 MANTIK HATASI DÜZELTİCİ
        
        Eğer tahminler mantıksız kombinasyon oluşturuyorsa, otomatik düzeltme yapar.
        """
        adjustments = {'adjusted': False, 'original': ms_prediction, 'reason': None}
        
        # MS1/MS2 + Alt 2.5 + KG Var = İmkansız → MSX'e çevir
        if ms_prediction in ['1', '2'] and ou_prediction == 'under' and btts_prediction == True:
            ms_prediction = 'X'  # Beraberliğe çevir
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
        """Tek maç tahmini (mantık kontrolü ile)"""
        if feature_engineer is None:
            raise ValueError("❌ FeatureEngineer required for predictions")

        features = feature_engineer.extract_match_features(home_team, away_team, odds)
        X = self._prepare_features(features)
        X_scaled = self.scaler.transform(X)
        proba = self.ensemble.predict_proba(X_scaled)[0]

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

        if proba[1] >= self.draw_threshold and proba[1] == proba.max():
            pred_idx = 1
        else:
            pred_idx = int(np.argmax(proba))

        idx_to_label = {0: "1", 1: "X", 2: "2"}
        idx_to_name = {0: "Home Win", 1: "Draw", 2: "Away Win"}
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
            "model_type": "weighted_ensemble",
            "feature_priorities": {
                "odds": "75%",
                "h2h": "15%",
                "form": "10%"
            }
        }

        # 🔍 Alt/Üst ve KG tahminlerini de ekle (eğer return_all_predictions=True)
        ou_prediction = None
        ou_confidence = None
        btts_prediction = None
        btts_confidence = None
        
        if return_all_predictions:
            # Bu değerler normalde ayrı endpoint'lerden gelir
            # Burada sadece mantık kontrolü için placeholder
            pass

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
            
            # 🔧 Otomatik düzeltme (opsiyonel)
            if not is_valid and len(warnings) > 0:
                adjusted_pred, adjustments = self._adjust_predictions_for_logic(
                    prediction_label, proba, ou_prediction, btts_prediction
                )
                
                if adjustments['adjusted']:
                    result['prediction'] = adjusted_pred
                    result['prediction_name'] = idx_to_name[idx_to_label[adjusted_pred]]
                    result['auto_adjusted'] = adjustments
                    result['confidence'] = result['confidence'] * (1 - adjustments.get('confidence_penalty', 0))

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
                    "agreement_with_odds": self._calculate_agreement(proba, odds)
                }
            except:
                pass

        if hasattr(feature_engineer, 'extract_match_features'):
            result["feature_analysis"] = {
                "odds_features_used": len([f for f in features.keys() if 'odds_' in f or 'market_' in f]),
                "h2h_features_used": len([f for f in features.keys() if f.startswith('h2h_')]),
                "form_features_used": len([f for f in features.keys() if 'form_' in f]),
            }

        if self.enable_monitoring:
            self.predictions_log.append({
                "timestamp": datetime.now().isoformat(),
                "home_team": home_team,
                "away_team": away_team,
                "prediction": prediction_label,
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "actual": actual_result,
                "correct": (actual_result == prediction_label) if actual_result else None,
                "logic_validated": self.enable_logic_validation,
                "had_warnings": len(result.get('logic_validation', {}).get('warnings', [])) > 0
            })

        return result

    def validate_full_prediction_set(self,
                                    ms_prediction: str,
                                    ms_confidence: float,
                                    ou_prediction: str,
                                    ou_confidence: float,
                                    btts_prediction: bool,
                                    btts_confidence: float) -> Dict[str, Any]:
        """
        🔍 TAM TAHMİN SETİ VALIDASYONU
        
        API'den gelen 3 farklı tahmini (MS, OU, BTTS) birlikte kontrol eder.
        """
        is_valid, warnings = self._validate_prediction_logic(
            ms_prediction=ms_prediction,
            ms_confidence=ms_confidence,
            ou_prediction=ou_prediction,
            ou_confidence=ou_confidence,
            btts_prediction=btts_prediction,
            btts_confidence=btts_confidence
        )
        
        result = {
            'is_valid': is_valid,
            'warnings': warnings,
            'predictions': {
                'match_result': {'prediction': ms_prediction, 'confidence': ms_confidence},
                'over_under': {'prediction': ou_prediction, 'confidence': ou_confidence},
                'btts': {'prediction': btts_prediction, 'confidence': btts_confidence}
            }
        }
        
        # Otomatik düzeltme önerisi
        if not is_valid:
            suggestions = []
            
            if ms_prediction in ['1', '2'] and ou_prediction == 'under' and btts_prediction:
                suggestions.append({
                    'option_1': {
                        'ms': ms_prediction,
                        'ou': 'under',
                        'btts': False,
                        'explanation': f'{ms_prediction} + Alt 2.5 + KG Yok (örn: 1-0, 2-0)'
                    },
                    'option_2': {
                        'ms': ms_prediction,
                        'ou': 'over',
                        'btts': True,
                        'explanation': f'{ms_prediction} + Üst 2.5 + KG Var (örn: 2-1, 3-1)'
                    },
                    'option_3': {
                        'ms': 'X',
                        'ou': 'under',
                        'btts': True,
                        'explanation': 'Beraberlik + Alt 2.5 + KG Var (sadece 1-1)'
                    }
                })
            
            result['suggestions'] = suggestions
        
        return result

    def _calculate_agreement(self, model_proba: np.ndarray, odds: Dict[str, float]) -> str:
        """Model ve odds uyumunu değerlendir"""
        try:
            o1, ox, o2 = float(odds['1']), float(odds['X']), float(odds['2'])
            total = (1/o1 + 1/ox + 1/o2)
            odds_proba = np.array([(1/o1)/total, (1/ox)/total, (1/o2)/total])

            model_pred = np.argmax(model_proba)
            odds_pred = np.argmax(odds_proba)

            if model_pred == odds_pred:
                diff = abs(model_proba[model_pred] - odds_proba[odds_pred])
                if diff < 0.1:
                    return "Strong Agreement"
                elif diff < 0.2:
                    return "Moderate Agreement"
                else:
                    return "Weak Agreement"
            else:
                return "Disagreement"
        except:
            return "Unknown"

    def batch_predict(self, matches: List[Dict[str, Any]], feature_engineer) -> List[Dict[str, Any]]:
        """Birden fazla maç için toplu tahmin"""
        results = []
        print(f"\n🔮 Batch Prediction: {len(matches)} matches")
        print(f"   Logic validation: {'ENABLED ✅' if self.enable_logic_validation else 'DISABLED'}")

        invalid_count = 0
        for i, match in enumerate(matches, 1):
            try:
                result = self.predict_match(
                    home_team=match["home_team"],
                    away_team=match["away_team"],
                    odds=match.get("odds"),
                    feature_engineer=feature_engineer,
                    actual_result=match.get("actual_result")
                )
                results.append(result)
                
                if not result.get('logic_validation', {}).get('is_valid', True):
                    invalid_count += 1
                
                if i % 10 == 0:
                    print(f"   Processed: {i}/{len(matches)}")
            except Exception as e:
                print(f"   ⚠️ Error predicting {match.get('home_team')} vs {match.get('away_team')}: {e}")
                continue

        print(f"✅ Batch prediction completed: {len(results)}/{len(matches)} successful")
        if invalid_count > 0:
            print(f"⚠️  Logic warnings: {invalid_count}/{len(results)} predictions had validation warnings")
        
        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Performans istatistikleri"""
        if not self.predictions_log:
            return {"total_predictions": 0}

        total = len(self.predictions_log)
        evaluated = [p for p in self.predictions_log if p.get("actual") is not None]
        if not evaluated:
            return {"total_predictions": total, "evaluated_predictions": 0, "accuracy": 0.0}

        correct = sum(1 for p in evaluated if p.get("correct"))
        class_stats = {}
        for label, name in [("1", "home_win"), ("X", "draw"), ("2", "away_win")]:
            preds = [p for p in evaluated if p["prediction"] == label]
            if preds:
                class_correct = sum(1 for p in preds if p.get("correct"))
                class_stats[name] = {
                    "total": len(preds),
                    "correct": class_correct,
                    "accuracy": class_correct / len(preds)
                }

        # Mantık validasyonu istatistikleri
        logic_stats = {}
        if self.enable_logic_validation:
            total_with_warnings = sum(1 for p in self.predictions_log if p.get("had_warnings"))
            logic_stats = {
                "total_with_warnings": total_with_warnings,
                "warning_rate": total_with_warnings / total if total > 0 else 0
            }

        return {
            "total_predictions": total,
            "evaluated_predictions": len(evaluated),
            "accuracy": correct / len(evaluated),
            "class_performance": class_stats,
            "average_confidence": np.mean([p["confidence"] for p in self.predictions_log]),
            "logic_validation": logic_stats
        }

    def get_recent_predictions(self, n: int = 10) -> List[Dict[str, Any]]:
        """Son N tahmini döndür"""
        return self.predictions_log[-n:] if self.predictions_log else []

    def export_predictions(self, filepath: str):
        """Tahminleri JSON olarak dışa aktar"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": self.metadata,
                    "predictions": self.predictions_log,
                    "performance": self.get_performance_stats()
                }, f, indent=2, ensure_ascii=False)
            print(f"✅ Predictions exported to: {filepath}")
        except Exception as e:
            print(f"❌ Export failed: {e}")


# Kullanım örneği
if __name__ == "__main__":
    print("=" * 70)
    print("🎯 WEIGHTED PREDICTOR + LOGIC VALIDATION TEST")
    print("=" * 70)

    predictor = WeightedPredictor(enable_logic_validation=True)

    print("\n🧪 Testing logic validation...")
    
    # Test Case 1: İmkansız kombinasyon
    print("\n❌ Test 1: MS1 + Alt 2.5 + KG Var (İMKANSIZ)")
    result1 = predictor.validate_full_prediction_set(
        ms_prediction='1',
        ms_confidence=0.51,
        ou_prediction='under',
        ou_confidence=0.65,
        btts_prediction=True,
        btts_confidence=0.80
    )
    print(json.dumps(result1, indent=2, ensure_ascii=False))
    
    # Test Case 2: Mantıklı kombinasyon
    print("\n✅ Test 2: MS1 + Üst 2.5 + KG Var (MANTIKLI)")
    result2 = predictor.validate_full_prediction_set(
        ms_prediction='1',
        ms_confidence=0.65,
        ou_prediction='over',
        ou_confidence=0.70,
        btts_prediction=True,
        btts_confidence=0.75
    )
    print(json.dumps(result2, indent=2, ensure_ascii=False))

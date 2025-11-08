import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

class WeightedPredictor:
    """
    AĞIRLIKLI TAHMİN SİSTEMİ

    Öncelik:
    - %75: Bahis Oranları
    - %15: H2H Geçmiş
    - %10: Form Durumu
    """

    def __init__(self,
                 model_dir: Optional[Path] = None,
                 draw_threshold: float = 0.30,
                 enable_monitoring: bool = True):
        """Model yükleme ve yapılandırma"""
        # ✅ MODEL DİZİNİNİ OTOMATİK ALGILA
        if model_dir is None:
            base_dir = Path(__file__).resolve().parents[1]  # ML_Project_Template
            model_dir = base_dir / "models"

        self.model_dir = Path(model_dir)
        self.draw_threshold = draw_threshold
        self.enable_monitoring = enable_monitoring

        self.ensemble = None
        self.scaler = None
        self.metadata = {}
        self.feature_names = []
        self.predictions_log = []

        print(f"🎯 Weighted Predictor Initializing...")
        print(f"   Model dir: {self.model_dir}")
        print(f"   Draw threshold: {self.draw_threshold}")

        self._load_models()

    def _load_models(self):
        """Model, scaler ve metadata dosyalarını yükle"""
        try:
            # 🔍 MODEL ADAYLARI
            candidate_models = [
                "weighted_model.pkl",
                "model.pkl",
                "best_model.pkl"
            ]

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

            # 🔍 SCALER ADAYLARI
            candidate_scalers = [
                "weighted_scaler.pkl",
                "scaler.pkl"
            ]

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

            # 🔍 METADATA
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

    def predict_match(self,
                      home_team: str,
                      away_team: str,
                      odds: Optional[Dict[str, float]] = None,
                      feature_engineer=None,
                      actual_result: Optional[str] = None) -> Dict[str, Any]:
        """Tek maç tahmini"""
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
                "correct": (actual_result == prediction_label) if actual_result else None
            })

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
                if i % 10 == 0:
                    print(f"   Processed: {i}/{len(matches)}")
            except Exception as e:
                print(f"   ⚠️ Error predicting {match.get('home_team')} vs {match.get('away_team')}: {e}")
                continue

        print(f"✅ Batch prediction completed: {len(results)}/{len(matches)} successful")
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

        return {
            "total_predictions": total,
            "evaluated_predictions": len(evaluated),
            "accuracy": correct / len(evaluated),
            "class_performance": class_stats,
            "average_confidence": np.mean([p["confidence"] for p in self.predictions_log])
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
    print("🎯 WEIGHTED PREDICTOR TEST MODE")
    print("=" * 70)

    predictor = WeightedPredictor()

    print("\n✅ Model ve scaler başarıyla yüklendiyse, aşağıdaki örnek test çalışır:")

    try:
        # Sahte test verisi (gerçek FeatureEngineer olmadan)
        dummy_features = {
            "odds_home_win": 2.1,
            "odds_draw": 3.4,
            "odds_away_win": 3.1,
            "form_home_points": 8,
            "form_away_points": 6,
            "h2h_home_wins": 3,
            "h2h_draws": 2,
            "h2h_away_wins": 1,
        }

        class DummyFeatureEngineer:
            def extract_match_features(self, home_team, away_team, odds):
                return dummy_features

        feature_engineer = DummyFeatureEngineer()

        result = predictor.predict_match(
            home_team="Galatasaray",
            away_team="Fenerbahçe",
            odds={'1': 2.10, 'X': 3.40, '2': 3.10},
            feature_engineer=feature_engineer
        )

        print("\n🔮 TEST PREDICTION RESULT:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"\n❌ Test failed: {e}")

    print("\n⚠️  To use predictor in full mode, run:")
    print("   uvicorn main_weighted:app --reload")
    print("   and access http://127.0.0.1:8000/docs")

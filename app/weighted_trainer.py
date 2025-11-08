import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import joblib
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

# ⚠️ MODEL BOYUTU OPTİMİZASYONU
# Hedef: Maksimum 24MB

MODEL_SIZE_CONFIG = {
    'max_depth': 5,              # Daha sığ ağaçlar (7 → 5)
    'n_estimators': 200,         # Daha az ağaç (500 → 200)
    'max_leaves': 31,            # Daha az yaprak
    'subsample': 0.7,            # Daha az veri örnekleme
    'colsample_bytree': 0.7,     # Daha az özellik örnekleme
    'use_best_model_only': True, # Sadece en iyi modeli kaydet
}

# AĞIRLIKLANDIRMA STRATEJİSİ
FEATURE_WEIGHTS = {
    'odds': 0.75,      # %75 - Bahis oranları
    'h2h': 0.15,       # %15 - Head-to-head
    'form': 0.10,      # %10 - Form durumu
}

def find_datasets():
    """Dataset'leri bul"""
    base_paths = [
        Path(__file__).resolve().parent.parent / "data" / "raw",
        Path(__file__).resolve().parent.parent / "data",
        Path(__file__).resolve().parent.parent,
    ]
    csv_path = None

    for base in base_paths:
        if (base / "merged_all_uniform_clean.csv").exists():
            csv_path = base / "merged_all_uniform_clean.csv"
            break
        if (base / "merged_all.csv").exists():
            csv_path = base / "merged_all.csv"
            break
    
    return csv_path

def clean_team_names(df):
    """Takım isimlerini temizle"""
    for col in ["home_team", "away_team"]:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(["", " ", "nan", "None", "NaN"], np.nan)
        df[col] = df[col].fillna("Unknown")
    return df

def prepare_odds(row):
    """Bahis oranlarını hazırla"""
    odds = {}
    
    odds_cols = {
        '1': ['odds_1', 'odds_home', 'home_odds'],
        'X': ['odds_x', 'odds_X', 'odds_draw', 'draw_odds'],
        '2': ['odds_2', 'odds_away', 'away_odds']
    }
    
    for key, possible_cols in odds_cols.items():
        value = None
        for col in possible_cols:
            if col in row and pd.notna(row[col]):
                try:
                    value = float(row[col])
                    if value > 0:
                        break
                except (ValueError, TypeError):
                    continue
        odds[key] = value
    
    if any(v is None or v <= 0 for v in odds.values()):
        return None
    
    return odds

def apply_feature_weights(X, feature_names):
    """
    ÖZELLİKLERE AĞIRLIK UYGULA
    
    %75 - Bahis oranları
    %15 - H2H
    %10 - Form
    """
    X_weighted = X.copy()
    
    for i, name in enumerate(feature_names):
        if any(key in name for key in ['odds_', 'market_', 'favorite', 'value', 'draw_market', 'clear_favorite', 'balanced_match']):
            X_weighted[:, i] = X_weighted[:, i] * FEATURE_WEIGHTS['odds']
            
        elif name.startswith('h2h_'):
            X_weighted[:, i] = X_weighted[:, i] * FEATURE_WEIGHTS['h2h']
            
        elif 'form_' in name or 'momentum' in name:
            X_weighted[:, i] = X_weighted[:, i] * FEATURE_WEIGHTS['form']
    
    return X_weighted

def get_model_size(obj):
    """Model boyutunu hesapla (MB)"""
    import tempfile
    try:
        # Windows-safe temporary file handling
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.pkl')
        os.close(tmp_fd)  # Close file descriptor immediately
        
        joblib.dump(obj, tmp_path, compress=3)
        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        
        # Clean up
        try:
            os.unlink(tmp_path)
        except:
            pass
            
        return size_mb
    except Exception as e:
        print(f"   ⚠️ Could not measure size: {e}")
        return 0.0

def train_weighted_model(test_size=0.2, cv_folds=5):
    """
    BOYUT-OPTİMİZE AĞIRLIKLI MODEL EĞİTİMİ
    
    Hedef: Maksimum 24MB model
    
    Öncelik:
    - %75 Bahis Oranları
    - %15 H2H Geçmiş
    - %10 Form Durumu
    """
    try:
        start_time = datetime.now()
        
        print("="*70)
        print("🎯 OPTIMIZED WEIGHTED MODEL TRAINING (MAX 24MB)")
        print("="*70)
        print("📊 Feature Priorities:")
        print(f"   • Odds Features:    {FEATURE_WEIGHTS['odds']*100:.0f}% weight")
        print(f"   • H2H Features:     {FEATURE_WEIGHTS['h2h']*100:.0f}% weight")
        print(f"   • Form Features:    {FEATURE_WEIGHTS['form']*100:.0f}% weight")
        print("\n🎯 Size Optimizations:")
        print(f"   • Max Depth:        {MODEL_SIZE_CONFIG['max_depth']}")
        print(f"   • N Estimators:     {MODEL_SIZE_CONFIG['n_estimators']}")
        print(f"   • Max Leaves:       {MODEL_SIZE_CONFIG['max_leaves']}")
        print("="*70)
        
        # Dataset yükleme
        csv_path = find_datasets()
        if csv_path is None:
            print("❌ No dataset found.")
            return False

        print(f"\n📂 Loading dataset: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
        df = clean_team_names(df)

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.sort_values('date').reset_index(drop=True)

        print(f"✅ Dataset loaded: {len(df)} matches")

        # Feature Engineering (Weighted)
        print("\n⚙️ Initializing Weighted Feature Engineering...")
        from app.weighted_feature_engineer import WeightedFeatureEngineer
        feature_engineer = WeightedFeatureEngineer(df)

        # Maçları hazırla
        print("\n📊 Preparing matches...")
        matches = []
        for idx, row in df.iterrows():
            odds = prepare_odds(row)
            matches.append((idx, row["home_team"], row["away_team"], row["result"], odds))
        
        print(f"✅ Prepared {len(matches)} matches")
        
        # Feature extraction
        print(f"\n🚀 Extracting weighted features...")
        X, y = feature_engineer.extract_features_batch(matches, show_progress=True)

        if len(X) == 0:
            print("❌ No valid features extracted!")
            return False

        print(f"\n✅ Features extracted: {X.shape}")
        
        # Feature isimleri
        feature_names = feature_engineer.get_feature_names()
        
        # Train-test split
        print(f"\n📊 Splitting dataset (test_size={test_size})...")
        if 'date' in df.columns:
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            print(f"   Using TEMPORAL split")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            print(f"   Using STRATIFIED split")
        
        print(f"   Training set: {len(X_train)}")
        print(f"   Test set: {len(X_test)}")

        # ⭐ AĞIRLIKLARI UYGULA
        print(f"\n⚖️ Applying feature weights...")
        X_train_weighted = apply_feature_weights(X_train, feature_names)
        X_test_weighted = apply_feature_weights(X_test, feature_names)

        # Scaling
        print("\n⚖️ Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_weighted)
        X_test_scaled = scaler.transform(X_test_weighted)

        # Class weights (beraberliğe öncelik)
        class_weights = {
            0: 1.0,   # Home win
            1: 1.5,   # Draw (50% daha fazla ağırlık)
            2: 1.0,   # Away win
        }

        # 🎯 BOYUT-OPTİMİZE MODELLER
        print("\n🤖 Creating size-optimized ensemble models...")
        
        cfg = MODEL_SIZE_CONFIG
        
        models = [
            ("xgboost", xgb.XGBClassifier(
                n_estimators=cfg['n_estimators'],
                max_depth=cfg['max_depth'],
                learning_rate=0.05,  # Daha yüksek learning rate
                subsample=cfg['subsample'],
                colsample_bytree=cfg['colsample_bytree'],
                scale_pos_weight=1.5,
                random_state=42,
                eval_metric='mlogloss',
                n_jobs=-1,
                verbosity=0
            )),
            ("lightgbm", lgb.LGBMClassifier(
                n_estimators=cfg['n_estimators'],
                max_depth=cfg['max_depth'],
                learning_rate=0.05,
                num_leaves=cfg['max_leaves'],
                class_weight=class_weights,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )),
        ]

        # 🏋️ Model eğitimi
        print(f"\n🏋️ Training size-optimized models...")
        individual_scores = {}
        trained_models = []
        
        for name, model in models:
            print(f"\n   {'='*50}")
            print(f"   Training: {name}")
            print(f"   {'='*50}")
            
            model.fit(X_train_scaled, y_train)
            
            # Boyut kontrolü
            model_size = get_model_size(model)
            print(f"   💾 Model size: {model_size:.2f} MB")
            
            if model_size > 15:  # Tek model 15MB'ı geçmesin
                print(f"   ⚠️ Model too large, skipping...")
                continue
            
            # Predictions
            test_pred = model.predict(X_test_scaled)
            test_proba = model.predict_proba(X_test_scaled)
            
            # Metrics
            test_acc = accuracy_score(y_test, test_pred)
            test_ll = log_loss(y_test, test_proba)
            
            # Draw accuracy
            draw_mask = y_test == 1
            draw_acc = 0.0
            if draw_mask.sum() > 0:
                draw_acc = (test_pred[draw_mask] == 1).sum() / draw_mask.sum()
            
            individual_scores[name] = {
                'test_accuracy': test_acc,
                'test_logloss': test_ll,
                'draw_accuracy': draw_acc,
                'model_size_mb': model_size
            }
            
            print(f"   ✓ Test Acc:     {test_acc:.4f}")
            print(f"   ✓ Draw Acc:     {draw_acc:.4f}")
            
            trained_models.append((name, model))

        if not trained_models:
            print("\n❌ No models trained successfully!")
            return False

        # En iyi modeli seç
        print(f"\n🏆 Selecting best model...")
        
        best_score = 0
        best_model = None
        best_name = None
        
        for name, model in trained_models:
            score = (individual_scores[name]['test_accuracy'] * 0.7 + 
                    individual_scores[name]['draw_accuracy'] * 0.3)
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
        
        print(f"   🥇 Best model: {best_name}")
        print(f"   📊 Test Acc: {individual_scores[best_name]['test_accuracy']:.4f}")
        print(f"   🎯 Draw Acc: {individual_scores[best_name]['draw_accuracy']:.4f}")
        print(f"   💾 Size: {individual_scores[best_name]['model_size_mb']:.2f} MB")

        # Final evaluation
        final_pred = best_model.predict(X_test_scaled)
        final_proba = best_model.predict_proba(X_test_scaled)
        
        ensemble_acc = accuracy_score(y_test, final_pred)
        ensemble_ll = log_loss(y_test, final_proba)
        
        # Draw analysis
        draw_mask = y_test == 1
        draw_pred_mask = final_pred == 1
        
        if draw_mask.sum() > 0:
            draw_recall = ((y_test == 1) & (final_pred == 1)).sum() / draw_mask.sum()
        else:
            draw_recall = 0.0
            
        if draw_pred_mask.sum() > 0:
            draw_precision = ((y_test == 1) & (final_pred == 1)).sum() / draw_pred_mask.sum()
        else:
            draw_precision = 0.0

        print(f"\n✅ Final Model Performance:")
        print(f"   Test Accuracy:  {ensemble_acc:.4f}")
        print(f"   Test Log Loss:  {ensemble_ll:.4f}")
        print(f"   Draw Recall:    {draw_recall:.4f}")
        print(f"   Draw Precision: {draw_precision:.4f}")

        # Classification report
        print(f"\n📊 Classification Report:")
        print(classification_report(y_test, final_pred,
                                  target_names=['Home Win', 'Draw', 'Away Win'],
                                  digits=4))

        # Model kaydetme
        print(f"\n💾 Saving optimized model...")
        model_dir = Path(__file__).parent.parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "weighted_model.pkl"
        scaler_path = model_dir / "weighted_scaler.pkl"
        
        joblib.dump(best_model, model_path, compress=3)  # Sıkıştırma seviyesi 3
        joblib.dump(scaler, scaler_path, compress=3)
        
        # Boyut kontrolü
        final_model_size = get_model_size(best_model)
        final_scaler_size = get_model_size(scaler)
        total_size = final_model_size + final_scaler_size
        
        print(f"   ✓ Model → {model_path}")
        print(f"     Size: {final_model_size:.2f} MB")
        print(f"   ✓ Scaler → {scaler_path}")
        print(f"     Size: {final_scaler_size:.2f} MB")
        print(f"   📦 Total: {total_size:.2f} MB")
        
        if total_size > 24:
            print(f"   ⚠️ WARNING: Total size ({total_size:.2f} MB) exceeds 24MB target!")
        else:
            print(f"   ✅ SUCCESS: Model fits within 24MB limit!")

        # Metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'model_type': f'weighted_{best_name}_optimized',
            'feature_weights': FEATURE_WEIGHTS,
            'size_config': MODEL_SIZE_CONFIG,
            'n_features': int(X.shape[1]),
            'feature_names': feature_names,
            'training_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'class_weights': {int(k): float(v) for k, v in class_weights.items()},
            'test_accuracy': float(ensemble_acc),
            'test_logloss': float(ensemble_ll),
            'draw_recall': float(draw_recall),
            'draw_precision': float(draw_precision),
            'model_size_mb': float(final_model_size),
            'scaler_size_mb': float(final_scaler_size),
            'total_size_mb': float(total_size),
            'individual_scores': {
                name: {k: float(v) for k, v in scores.items()}
                for name, scores in individual_scores.items()
            }
        }
        
        metadata_path = model_dir / 'weighted_model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✓ Metadata → {metadata_path}")

        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"\n{'='*70}")
        print("🎉 OPTIMIZED TRAINING COMPLETED!")
        print(f"{'='*70}")
        print(f"⏱️  Duration: {duration:.1f}s")
        print(f"📊 Test Accuracy: {ensemble_acc:.4f}")
        print(f"🎯 Draw Recall: {draw_recall:.4f}")
        print(f"💾 Total Size: {total_size:.2f} MB / 24 MB")
        print(f"{'='*70}")

        return True

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    train_weighted_model()
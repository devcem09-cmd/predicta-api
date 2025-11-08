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
            # Bahis oranı özellikleri - %75 ağırlık
            X_weighted[:, i] = X_weighted[:, i] * FEATURE_WEIGHTS['odds']
            
        elif name.startswith('h2h_'):
            # H2H özellikleri - %15 ağırlık
            X_weighted[:, i] = X_weighted[:, i] * FEATURE_WEIGHTS['h2h']
            
        elif 'form_' in name or 'momentum' in name:
            # Form özellikleri - %10 ağırlık
            X_weighted[:, i] = X_weighted[:, i] * FEATURE_WEIGHTS['form']
    
    return X_weighted

def train_weighted_model(test_size=0.2, cv_folds=5):
    """
    AĞIRLIKLI MODEL EĞİTİMİ
    
    Öncelik:
    - %75 Bahis Oranları
    - %15 H2H Geçmiş
    - %10 Form Durumu
    """
    try:
        start_time = datetime.now()
        
        print("="*70)
        print("🎯 WEIGHTED MODEL TRAINING")
        print("="*70)
        print("📊 Feature Priorities:")
        print(f"   • Odds Features:    {FEATURE_WEIGHTS['odds']*100:.0f}% weight")
        print(f"   • H2H Features:     {FEATURE_WEIGHTS['h2h']*100:.0f}% weight")
        print(f"   • Form Features:    {FEATURE_WEIGHTS['form']*100:.0f}% weight")
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
        print(f"   Features: {X.shape[1]}")
        print(f"   Samples: {X.shape[0]}")
        
        # Feature isimleri
        feature_names = feature_engineer.get_feature_names()
        print(f"   Feature names: {len(feature_names)}")
        
        # Feature kategorilerini say
        odds_features = [f for f in feature_names if any(k in f for k in ['odds_', 'market_', 'favorite', 'value'])]
        h2h_features = [f for f in feature_names if f.startswith('h2h_')]
        form_features = [f for f in feature_names if 'form_' in f or 'momentum' in f]
        
        print(f"\n📈 Feature Breakdown:")
        print(f"   Odds features:  {len(odds_features)} ({len(odds_features)/len(feature_names)*100:.1f}%)")
        print(f"   H2H features:   {len(h2h_features)} ({len(h2h_features)/len(feature_names)*100:.1f}%)")
        print(f"   Form features:  {len(form_features)} ({len(form_features)/len(feature_names)*100:.1f}%)")
        print(f"   Other features: {len(feature_names) - len(odds_features) - len(h2h_features) - len(form_features)}")

        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"\n📊 Class distribution:")
        for cls, count in zip(unique, counts):
            label = ['Home Win (1)', 'Draw (X)', 'Away Win (2)'][cls]
            print(f"   {label}: {count} ({count/len(y)*100:.1f}%)")

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
        print(f"   Odds: x{FEATURE_WEIGHTS['odds']}")
        print(f"   H2H: x{FEATURE_WEIGHTS['h2h']}")
        print(f"   Form: x{FEATURE_WEIGHTS['form']}")
        
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
        
        print(f"\n⚖️ Class weights:")
        for cls, weight in class_weights.items():
            label = ['Home Win', 'Draw', 'Away Win'][cls]
            print(f"   {label}: {weight:.2f}")

        # Model oluşturma
        print("\n🤖 Creating ensemble models...")
        models = [
            ("xgboost", xgb.XGBClassifier(
                n_estimators=500,
                max_depth=7,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1.5,  # Draw boost
                random_state=42,
                eval_metric='mlogloss',
                n_jobs=-1,
                verbosity=0
            )),
            ("lightgbm", lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=7,
                learning_rate=0.03,
                num_leaves=50,
                class_weight=class_weights,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )),
            ("catboost", CatBoostClassifier(
                iterations=500,
                depth=7,
                learning_rate=0.03,
                class_weights=[1.0, 1.5, 1.0],  # Draw boost
                random_seed=42,
                verbose=False,
                thread_count=-1
            )),
            ("random_forest", RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=10,
                class_weight=class_weights,
                random_state=42,
                n_jobs=-1
            ))
        ]

        # Model eğitimi
        print(f"\n🏋️ Training individual models...")
        individual_scores = {}
        
        for name, model in models:
            print(f"\n   {'='*50}")
            print(f"   Training: {name}")
            print(f"   {'='*50}")
            
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            train_proba = model.predict_proba(X_train_scaled)
            test_proba = model.predict_proba(X_test_scaled)
            
            # Metrics
            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            train_ll = log_loss(y_train, train_proba)
            test_ll = log_loss(y_test, test_proba)
            
            # Class-specific accuracy
            draw_mask = y_test == 1
            draw_acc = 0.0
            if draw_mask.sum() > 0:
                draw_acc = (test_pred[draw_mask] == 1).sum() / draw_mask.sum()
            
            # CV
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                scoring='accuracy',
                n_jobs=-1
            )
            
            individual_scores[name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'train_logloss': train_ll,
                'test_logloss': test_ll,
                'draw_accuracy': draw_acc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"   ✓ Train Acc:    {train_acc:.4f}")
            print(f"   ✓ Test Acc:     {test_acc:.4f}")
            print(f"   ✓ Draw Acc:     {draw_acc:.4f}")
            print(f"   ✓ CV Acc:       {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            
            # Model kaydet
            model_dir = Path(__file__).parent.parent / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"model_{name}.pkl"
            joblib.dump(model, model_path)
            print(f"   💾 Saved → {model_path}")

        # Ensemble oluşturma
        print(f"\n🔗 Creating weighted ensemble...")
        
        # Ensemble weights (draw performance'a göre)
        weights = []
        for name, metrics in individual_scores.items():
            # Test accuracy + Draw accuracy kombinasyonu
            weight = (metrics['test_accuracy'] * 0.5) + (metrics['draw_accuracy'] * 0.5)
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        print(f"   Ensemble weights:")
        for (name, _), weight in zip(models, weights):
            print(f"   {name}: {weight:.3f}")
        
        ensemble = VotingClassifier(
            estimators=models,
            voting='soft',
            weights=weights,
            n_jobs=-1
        )
        ensemble.fit(X_train_scaled, y_train)

        # Ensemble evaluation
        print(f"\n📊 Evaluating ensemble...")
        ensemble_pred = ensemble.predict(X_test_scaled)
        ensemble_proba = ensemble.predict_proba(X_test_scaled)
        
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        ensemble_ll = log_loss(y_test, ensemble_proba)
        
        # Draw analysis
        draw_mask = y_test == 1
        draw_pred_mask = ensemble_pred == 1
        
        if draw_mask.sum() > 0:
            draw_recall = ((y_test == 1) & (ensemble_pred == 1)).sum() / draw_mask.sum()
        else:
            draw_recall = 0.0
            
        if draw_pred_mask.sum() > 0:
            draw_precision = ((y_test == 1) & (ensemble_pred == 1)).sum() / draw_pred_mask.sum()
        else:
            draw_precision = 0.0

        print(f"\n✅ Ensemble Performance:")
        print(f"   Test Accuracy:  {ensemble_acc:.4f}")
        print(f"   Test Log Loss:  {ensemble_ll:.4f}")
        print(f"   Draw Recall:    {draw_recall:.4f}")
        print(f"   Draw Precision: {draw_precision:.4f}")

        # Classification report
        print(f"\n📊 Classification Report:")
        print(classification_report(y_test, ensemble_pred,
                                  target_names=['Home Win', 'Draw', 'Away Win'],
                                  digits=4))

        # Confusion matrix
        print(f"\n🎯 Confusion Matrix:")
        cm = confusion_matrix(y_test, ensemble_pred)
        print("              Predicted")
        print("              1    X    2")
        for i, label in enumerate(['Actual 1', 'Actual X', 'Actual 2']):
            print(f"{label:10s} {cm[i][0]:5d} {cm[i][1]:4d} {cm[i][2]:4d}")

        # Model kaydetme
        print(f"\n💾 Saving models...")
        model_dir = Path(__file__).parent.parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "weighted_model.pkl"
        scaler_path = model_dir / "weighted_scaler.pkl"
        
        joblib.dump(ensemble, model_path)
        joblib.dump(scaler, scaler_path)
        
        print(f"   ✓ Ensemble → {model_path}")
        print(f"   ✓ Scaler → {scaler_path}")

        # Metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'model_type': 'weighted_ensemble',
            'feature_weights': FEATURE_WEIGHTS,
            'n_features': int(X.shape[1]),
            'feature_names': feature_names,
            'training_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'class_weights': {int(k): float(v) for k, v in class_weights.items()},
            'ensemble_weights': {name: float(weight) for name, weight in zip([m[0] for m in models], weights)},
            'test_accuracy': float(ensemble_acc),
            'test_logloss': float(ensemble_ll),
            'draw_recall': float(draw_recall),
            'draw_precision': float(draw_precision),
            'individual_scores': {
                name: {k: float(v) for k, v in scores.items()}
                for name, scores in individual_scores.items()
            },
            'feature_breakdown': {
                'odds_features': len(odds_features),
                'h2h_features': len(h2h_features),
                'form_features': len(form_features)
            }
        }
        
        metadata_path = model_dir / 'weighted_model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✓ Metadata → {metadata_path}")

        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"\n{'='*70}")
        print("🎉 WEIGHTED TRAINING COMPLETED!")
        print(f"{'='*70}")
        print(f"⏱️  Duration: {duration:.1f}s")
        print(f"📊 Test Accuracy: {ensemble_acc:.4f}")
        print(f"🎯 Draw Recall: {draw_recall:.4f}")
        print(f"🎯 Draw Precision: {draw_precision:.4f}")
        print(f"💾 Models saved to: {model_dir}")
        print(f"{'='*70}")

        return True

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_model():
    """Main training function that can be called from other modules"""
    print("🎯 Starting Weighted Model Training...")
    
    # Your existing training code here
    # ...
    
    # Save with protocol=4
    joblib.dump(
        model_data,
        model_path,
        compress=3,
        protocol=4  # Python 3.11 compatible
    )
    
    print("✅ Training completed")
    return model_path


if __name__ == "__main__":
    train_model()

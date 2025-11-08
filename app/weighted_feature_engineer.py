import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')

class WeightedFeatureEngineer:
    """
    AĞIRLIKLI ÖZELLİK ÇIKARIMI
    
    Öncelik Sırası:
    - %75: Bahis Oranları (Piyasa Beklentisi)
    - %15: H2H Geçmiş Performans
    - %10: Son Form Durumu
    """
    
    def __init__(self, df: pd.DataFrame, n_jobs: int = -1):
        self.df = df.copy()
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        
        # Veri temizleme
        self._clean_team_names()
        self._normalize_result_column()
        self._normalize_score_columns()
        
        # Tarih düzenleme
        self.has_date = False
        if 'date' in self.df.columns:
            try:
                self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
                if self.df['date'].notna().sum() > 0:
                    self.df = self.df.sort_values('date').reset_index(drop=True)
                    self.has_date = True
            except Exception as e:
                print(f"⚠️ Date parsing error: {e}")
        
        # Cache oluştur
        self._build_caches()
        self._convert_to_numpy()
        self._calculate_global_averages()
        
        print(f"📊 Weighted Feature Engineer initialized")
        print(f"   Matches: {len(self.df)}")
        print(f"   Priority: 75% Odds | 15% H2H | 10% Form")

    # ============================================================
    # VERİ TEMİZLEME VE HAZIRLIK
    # ============================================================
    
    def _clean_team_names(self):
        """Takım isimlerini temizle"""
        for col in ['home_team', 'away_team']:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()
                self.df[col] = self.df[col].replace(['', ' ', 'nan', 'None', 'NaN'], 'Unknown')
        
        before = len(self.df)
        self.df = self.df[
            (self.df['home_team'] != 'Unknown') & 
            (self.df['away_team'] != 'Unknown')
        ]
        if len(self.df) < before:
            print(f"⚠️ Removed {before - len(self.df)} invalid matches")
    
    def _normalize_result_column(self):
        """Result kolonunu standardize et"""
        result_candidates = ['result', 'Result', 'outcome', 'match_result', 'FTR']
        
        result_col = None
        for candidate in result_candidates:
            if candidate in self.df.columns:
                result_col = candidate
                break
        
        if result_col is None:
            raise ValueError("❌ No 'result' column found!")
        
        if result_col != 'result':
            self.df['result'] = self.df[result_col]
        
        self.df['result'] = self.df['result'].astype(str).str.strip().str.upper()
        
        result_mapping = {
            '1': '1', 'X': 'X', '2': '2',
            'H': '1', 'D': 'X', 'A': '2',
            'HOME': '1', 'DRAW': 'X', 'AWAY': '2',
        }
        
        self.df['result'] = self.df['result'].replace(result_mapping)
        
        valid_results = ['1', 'X', '2']
        before = len(self.df)
        self.df = self.df[self.df['result'].isin(valid_results)]
        
        if len(self.df) < before:
            print(f"⚠️ Removed {before - len(self.df)} invalid results")
    
    def _normalize_score_columns(self):
        """Skor kolonlarını standardize et"""
        score_mappings = {
            'home_score': ['home_score', 'HomeScore', 'home_goals', 'FTHG'],
            'away_score': ['away_score', 'AwayScore', 'away_goals', 'FTAG']
        }
        
        for target, candidates in score_mappings.items():
            if target not in self.df.columns:
                for candidate in candidates:
                    if candidate in self.df.columns:
                        self.df[target] = self.df[candidate]
                        break
        
        for col in ['home_score', 'away_score']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        before = len(self.df)
        self.df = self.df.dropna(subset=['home_score', 'away_score'])
        if len(self.df) < before:
            print(f"⚠️ Removed {before - len(self.df)} matches with missing scores")
    
    def _build_caches(self):
        """Hızlı erişim için cache'ler oluştur"""
        self.home_matches_idx = defaultdict(list)
        self.away_matches_idx = defaultdict(list)
        self.h2h_idx = defaultdict(list)
        
        self.df = self.df.reset_index(drop=True)
        
        for idx, row in self.df.iterrows():
            home = str(row['home_team'])
            away = str(row['away_team'])
            
            if home in ['Unknown', 'nan', 'None'] or away in ['Unknown', 'nan', 'None']:
                continue
                
            self.home_matches_idx[home].append(idx)
            self.away_matches_idx[away].append(idx)
            
            pair_key = tuple(sorted([home, away]))
            self.h2h_idx[pair_key].append(idx)
    
    def _convert_to_numpy(self):
        """Numpy array'e dönüştür"""
        self.home_scores = self.df['home_score'].values
        self.away_scores = self.df['away_score'].values
        self.results = self.df['result'].values
        self.home_teams = self.df['home_team'].values
        self.away_teams = self.df['away_team'].values

    def _calculate_global_averages(self):
        """Global ortalamaları hesapla"""
        self.global_avg_home_goals = self.df['home_score'].mean()
        self.global_avg_away_goals = self.df['away_score'].mean()

    # ============================================================
    # BAHIS ORANLARI ÖZELLİKLERİ (%75 AĞIRLIK)
    # ============================================================
    
    def _extract_odds_features(self, odds: Optional[Dict[str, float]]) -> Dict[str, float]:
        """
        BAHİS ORANLARI ÖZELLİKLERİ - EN ÖNEMLİ KISIM
        
        Bu özellikler %75 ağırlığa sahip olacak:
        - Implied probabilities (piyasa beklentileri)
        - Value indicators (değer tespiti)
        - Market confidence (piyasa güveni)
        - Odds spreads (oran farklılıkları)
        """
        default_features = {
            # Temel oranlar
            'odds_home': 2.50,
            'odds_draw': 3.20,
            'odds_away': 2.80,
            
            # Implied probabilities (normalize edilmiş)
            'odds_home_prob': 0.33,
            'odds_draw_prob': 0.33,
            'odds_away_prob': 0.33,
            
            # Market analysis
            'market_margin': 0.05,
            'market_confidence': 0.5,
            'favorite_odds': 2.50,
            'underdog_odds': 2.80,
            'odds_spread': 0.30,
            
            # Value indicators
            'home_value': 0.0,
            'draw_value': 0.0,
            'away_value': 0.0,
            
            # Draw-specific
            'draw_odds_level': 0.5,  # 0=düşük, 1=yüksek
            'draw_market_view': 0.33,
            
            # Certainty indicators
            'clear_favorite': 0.0,
            'balanced_match': 1.0,
        }
        
        if not odds or not isinstance(odds, dict):
            return default_features
        
        try:
            # Oranları al ve doğrula
            if not all(k in odds for k in ['1', 'X', '2']):
                return default_features
            
            if not all(odds[k] is not None and float(odds[k]) > 0 for k in ['1', 'X', '2']):
                return default_features
            
            odds_1 = float(odds['1'])
            odds_x = float(odds['X'])
            odds_2 = float(odds['2'])
            
            # Implied probabilities (normalize edilmiş)
            total_prob = (1/odds_1 + 1/odds_x + 1/odds_2)
            
            prob_1 = (1/odds_1) / total_prob
            prob_x = (1/odds_x) / total_prob
            prob_2 = (1/odds_2) / total_prob
            
            # Market margin (bookmaker komisyonu)
            market_margin = total_prob - 1.0
            
            # Market confidence (düşük margin = yüksek güven)
            market_confidence = 1.0 - min(market_margin, 0.15) / 0.15
            
            # Favorileri belirle
            favorite_odds = min(odds_1, odds_2)
            underdog_odds = max(odds_1, odds_2)
            odds_spread = underdog_odds - favorite_odds
            
            # Value indicators (expected vs market)
            historical_exp = [0.45, 0.27, 0.28]  # Tipik dağılım
            home_value = historical_exp[0] - prob_1
            draw_value = historical_exp[1] - prob_x
            away_value = historical_exp[2] - prob_2
            
            # Draw odds level
            if odds_x < 2.8:
                draw_odds_level = 0.0  # Düşük (beraberlik bekleniyor)
            elif odds_x > 3.6:
                draw_odds_level = 1.0  # Yüksek (beraberlik beklenmiyor)
            else:
                draw_odds_level = (odds_x - 2.8) / (3.6 - 2.8)
            
            # Clear favorite check
            clear_favorite = 1.0 if odds_spread > 1.0 else 0.0
            balanced_match = 1.0 if odds_spread < 0.4 else 0.0
            
            return {
                'odds_home': odds_1,
                'odds_draw': odds_x,
                'odds_away': odds_2,
                'odds_home_prob': prob_1,
                'odds_draw_prob': prob_x,
                'odds_away_prob': prob_2,
                'market_margin': market_margin,
                'market_confidence': market_confidence,
                'favorite_odds': favorite_odds,
                'underdog_odds': underdog_odds,
                'odds_spread': odds_spread,
                'home_value': home_value,
                'draw_value': draw_value,
                'away_value': away_value,
                'draw_odds_level': draw_odds_level,
                'draw_market_view': prob_x,
                'clear_favorite': clear_favorite,
                'balanced_match': balanced_match,
            }
            
        except (ValueError, ZeroDivisionError, TypeError):
            return default_features

    # ============================================================
    # H2H ÖZELLİKLERİ (%15 AĞIRLIK)
    # ============================================================
    
    def _extract_h2h_features(self, home_team: str, away_team: str, 
                          current_idx: int = None) -> Dict[str, float]:
    """
    HEAD-TO-HEAD ÖZELLİKLERİ
    
    Geçmiş karşılaşma verileri:
    - Son 10 maç istatistikleri
    - Kazanma oranları
    - Gol ortalamaları
    - Beraberlik eğilimi
    """
    pair_key = tuple(sorted([str(home_team), str(away_team)]))
    h2h_indices = self.h2h_idx.get(pair_key, [])
    
    # ✅ FIX: current_idx None ise tüm maçları al
    if current_idx is not None:
        h2h_indices = [i for i in h2h_indices if i < current_idx]
    
    # Son 10 maç
    h2h_indices = h2h_indices[-10:]
    
    if not h2h_indices:
        return {
            'h2h_matches': 0,
            'h2h_home_win_rate': 0.40,
            'h2h_draw_rate': 0.27,
            'h2h_away_win_rate': 0.33,
            'h2h_avg_home_goals': 1.3,
            'h2h_avg_away_goals': 1.1,
            'h2h_avg_total_goals': 2.4,
            'h2h_high_scoring': 0.0,
            'h2h_draw_tendency': 0.0,
        }
    
    indices = np.array(h2h_indices)
    home_is_home = self.home_teams[indices] == home_team
    
    home_goals = np.where(home_is_home, self.home_scores[indices], self.away_scores[indices])
    away_goals = np.where(home_is_home, self.away_scores[indices], self.home_scores[indices])
    
    results = self.results[indices]
    home_wins = np.sum(np.where(home_is_home, results == '1', results == '2'))
    draws = np.sum(results == 'X')
    away_wins = np.sum(np.where(home_is_home, results == '2', results == '1'))
    
    total = len(indices)
    avg_total = float(np.mean(home_goals + away_goals))
    
    return {
        'h2h_matches': total,
        'h2h_home_win_rate': float(home_wins / total),
        'h2h_draw_rate': float(draws / total),
        'h2h_away_win_rate': float(away_wins / total),
        'h2h_avg_home_goals': float(np.mean(home_goals)),
        'h2h_avg_away_goals': float(np.mean(away_goals)),
        'h2h_avg_total_goals': avg_total,
        'h2h_high_scoring': 1.0 if avg_total > 3.0 else 0.0,
        'h2h_draw_tendency': 1.0 if (draws / total) > 0.35 else 0.0,
    }
    # ============================================================
    # FORM ÖZELLİKLERİ (%10 AĞIRLIK)
    # ============================================================
    
    def _extract_form_features(self, team: str, current_idx: int = None, 
                           is_home: bool = True) -> Dict[str, float]:
    """
    SON FORM ÖZELLİKLERİ
    
    Son 5 maçtaki performans:
    - Puan ortalaması
    - Galibiyet oranı
    - Gol performansı
    """
    if is_home:
        all_indices = self.home_matches_idx.get(team, [])
    else:
        all_indices = self.away_matches_idx.get(team, [])
    
    # ✅ FIX: current_idx None ise tüm maçları al
    if current_idx is not None:
        indices = [i for i in all_indices if i < current_idx]
    else:
        indices = all_indices.copy()
    
    # Son 5 maç
    indices = indices[-5:]
    
    if not indices:
        return {
            'form_win_rate': 0.40,
            'form_points_per_game': 1.2,
            'form_avg_goals_scored': 1.2,
            'form_avg_goals_conceded': 1.2,
            'form_momentum': 0.5,
        }
    
    indices = np.array(indices)
    
    goals_scored = self.home_scores[indices] if is_home else self.away_scores[indices]
    goals_conceded = self.away_scores[indices] if is_home else self.home_scores[indices]
    
    results = self.results[indices]
    wins = np.sum(results == ('1' if is_home else '2'))
    draws = np.sum(results == 'X')
    points = wins * 3 + draws
    
    # Momentum (son maçların ağırlıklı ortalaması)
    weights = np.array([0.10, 0.15, 0.20, 0.25, 0.30])[:len(indices)]  # Son maça daha fazla ağırlık
    weights = weights / weights.sum()  # Normalize
    
    match_points = np.where(results == ('1' if is_home else '2'), 3, 
                            np.where(results == 'X', 1, 0))
    momentum = np.sum(match_points * weights) / 3.0  # Normalize to 0-1
    
    return {
        'form_win_rate': float(wins / len(indices)),
        'form_points_per_game': float(points / len(indices)),
        'form_avg_goals_scored': float(np.mean(goals_scored)),
        'form_avg_goals_conceded': float(np.mean(goals_conceded)),
        'form_momentum': float(momentum),
    }

    # ============================================================
    # ANA FEATURE EXTRACTION
    # ============================================================
    
    def extract_match_features(self, home_team: str, away_team: str,
                               odds: Optional[Dict[str, float]] = None,
                               current_date=None, current_idx: int = None) -> Dict[str, float]:
        """
        AĞIRLIKLI ÖZELLİK ÇIKARIMI
        
        Öncelik sırası:
        1. %75 - Bahis Oranları (18 özellik)
        2. %15 - H2H Geçmiş (9 özellik)
        3. %10 - Form Durumu (10 özellik)
        
        Toplam: 37 özellik
        """
        features = {}
        
        # 1. BAHİS ORANLARI ÖZELLİKLERİ (%75 ağırlık - 18 özellik)
        odds_features = self._extract_odds_features(odds)
        for key, value in odds_features.items():
            features[f'odds_{key}' if not key.startswith('odds_') else key] = value
        
        # 2. H2H ÖZELLİKLERİ (%15 ağırlık - 9 özellik)
        h2h_features = self._extract_h2h_features(home_team, away_team, current_idx)
        features.update(h2h_features)
        
        # 3. FORM ÖZELLİKLERİ (%10 ağırlık - 10 özellik)
        home_form = self._extract_form_features(home_team, current_idx, is_home=True)
        away_form = self._extract_form_features(away_team, current_idx, is_home=False)
        
        for key, value in home_form.items():
            features[f'home_{key}'] = value
        
        for key, value in away_form.items():
            features[f'away_{key}'] = value
        
        # 4. TÜREVLENMİŞ ÖZELLİKLER (kombinasyonlar)
        # Form farkları
        features['form_diff_momentum'] = home_form['form_momentum'] - away_form['form_momentum']
        features['form_diff_ppg'] = home_form['form_points_per_game'] - away_form['form_points_per_game']
        
        # Bahis + H2H kombinasyonu
        features['odds_h2h_alignment'] = abs(odds_features['odds_home_prob'] - h2h_features['h2h_home_win_rate'])
        
        # Bahis + Form kombinasyonu
        expected_from_odds = odds_features['odds_home_prob'] / (odds_features['odds_home_prob'] + odds_features['odds_away_prob'])
        expected_from_form = home_form['form_win_rate'] / (home_form['form_win_rate'] + away_form['form_win_rate'] + 0.01)
        features['odds_form_agreement'] = 1.0 - abs(expected_from_odds - expected_from_form)
        
        return features
    
    def extract_features_batch(self, matches: List[tuple], show_progress: bool = True) -> tuple:
        """Batch feature extraction"""
        X_list, y_list = [], []
        result_map = {"1": 0, "X": 1, "2": 2}
        
        successful = 0
        failed = 0
        
        for i, match_data in enumerate(matches):
            idx, home_team, away_team, result, odds = match_data
            
            if idx >= len(self.df) or result not in result_map:
                failed += 1
                continue
            
            try:
                features = self.extract_match_features(home_team, away_team, odds, None, idx)
                X_list.append(list(features.values()))
                y_list.append(result_map[result])
                successful += 1
            except Exception as e:
                failed += 1
                if show_progress and failed % 100 == 0:
                    print(f"⚠️ Errors: {failed}")
            
            if show_progress and (i + 1) % 1000 == 0:
                print(f"   Processed: {i + 1}/{len(matches)} (✓{successful} ✗{failed})", end='\r')
        
        if show_progress:
            print(f"\n   Total: {len(matches)} | Success: {successful} | Failed: {failed}")
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        if np.isnan(X).any():
            print(f"⚠️ Replacing {np.isnan(X).sum()} NaN values with 0")
            X = np.nan_to_num(X, 0)
        
        return X, y
    
    def get_feature_names(self) -> List[str]:
        """Feature isimlerini döndür"""
        dummy_features = self.extract_match_features("Team A", "Team B", None, None, 0)
        return list(dummy_features.keys())

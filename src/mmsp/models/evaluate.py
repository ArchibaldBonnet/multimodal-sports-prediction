import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from src.mmsp.models.baseline import train_sklearn_baseline
from src.mmsp.models.lstm import train_multimodal_lstm,MultimodalNet
from src.mmsp.nlp import get_text_features

def get_prepared_data():
    """Prépare le dataset canonique pour l'évaluation simultanée."""
    df = pd.read_parquet("data/dataset.parquet")
    
    conditions = [df['HomeScore'] < df['AwayScore'], df['HomeScore'] == df['AwayScore'], df['HomeScore'] > df['AwayScore']]
    y = torch.tensor(np.select(conditions, [0, 1, 2], default=np.nan), dtype=torch.long)
    
    features = [
        'Home_AvgScored_5', 'Home_AvgConceded_5', 'Home_Points_5', 
        'Away_AvgScored_5', 'Away_AvgConceded_5', 'Away_Points_5',
        'Home_Shots_5', 'Home_ShotsTarget_5', 'Home_Corners_5', 
        'Away_Shots_5', 'Away_ShotsTarget_5', 'Away_Corners_5'
    ]
    X_tab = StandardScaler().fit_transform(df[features])
    X_tab = torch.tensor(X_tab, dtype=torch.float32).unsqueeze(1)
    X_text = get_text_features(df)
    
    # Séparation temporelle stricte (80/20) imposée par la configuration du projet
    split_idx = int(len(df) * 0.8)
    return (
        X_tab[:split_idx], X_tab[split_idx:],
        X_text[:split_idx], X_text[split_idx:],
        y[:split_idx], y[split_idx:]
    )

def run_evaluation():
    """Phase 6 : Comparaison finale du Log-loss sur la même séparation temporelle."""
    print("Récupération des données canoniques...")
    X_tab_train, X_tab_test, X_text_train, X_text_test, y_train, y_test = get_prepared_data()
    
    print("\n1. Entraînement de la Sklearn Baseline (Convergence automatique)...")
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        sklearn_model = train_sklearn_baseline()
        df = pd.read_parquet("data/dataset.parquet")
        y_true = y_test.numpy()
        features = ['Home_AvgScored_5', 'Home_AvgConceded_5', 'Home_Points_5', 'Away_AvgScored_5', 'Away_AvgConceded_5', 'Away_Points_5']
        X_sk_test = df[features].iloc[int(len(df) * 0.8):]
        from sklearn.metrics import log_loss
        sklearn_loss = log_loss(y_true, sklearn_model.predict_proba(X_sk_test))

    print("2. Entraînement du LSTM (Stats-only sur 150 epochs)...")
    lstm_stats_loss = train_multimodal_lstm(X_tab_train, X_tab_test, X_text_train, X_text_test, y_train, y_test, use_text=False)
    
    print("3. Entraînement du Multimodal LSTM (Stats + Texte sur 150 epochs)...")
    lstm_multi_loss = train_multimodal_lstm(X_tab_train, X_tab_test, X_text_train, X_text_test, y_train, y_test, use_text=True)
    
    print("\n" + "="*50)
    print("🏆 RÉSULTATS FINAUX (PHASE 6) : LOG-LOSS")
    print("="*50)
    print(f"1. Sklearn Baseline          : {sklearn_loss:.4f}")
    print(f"2. LSTM (Stats-only)         : {lstm_stats_loss:.4f}")
    print(f"3. LSTM (Multimodal)         : {lstm_multi_loss:.4f}")
    print("="*50)
    print("Rappel : Plus le score est proche de 0, meilleures sont les probabilités.")
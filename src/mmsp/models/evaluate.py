import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from src.mmsp.models.baseline import train_sklearn_baseline
from src.mmsp.models.lstm import MultimodalNet
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

def train_eval_lstm(X_tab_train, X_tab_test, X_text_train, X_text_test, y_train, y_test, use_text=True):
    if not use_text:
        X_text_train = torch.zeros((X_text_train.shape[0], 1))
        X_text_test = torch.zeros((X_text_test.shape[0], 1))
        text_dim = 1
    else:
        text_dim = X_text_train.shape[1]

    model = MultimodalNet(tabular_dim=12, text_dim=text_dim)
    criterion = nn.CrossEntropyLoss()
    
    # Ajout du Weight Decay (régularisation L2) pour contrer l'overfitting
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-3)
    
    best_test_loss = float('inf')
    patience = 15          # Nombre d'époques tolérées sans amélioration
    patience_counter = 0
    best_weights = None

    for epoch in range(150):
        # 1. Apprentissage
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tab_train, X_text_train), y_train)
        loss.backward()
        optimizer.step()
        
        # 2. Évaluation
        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(X_tab_test, X_text_test), y_test).item()
            
        # 3. Early Stopping & Checkpoint
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            best_weights = model.state_dict().copy() # Sauvegarde des meilleurs poids
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # Arrêt anticipé avant que la courbe orange ne s'envole
                break

    # Restauration des meilleurs poids trouvés avant le surapprentissage
    if best_weights:
        model.load_state_dict(best_weights)
        
    return best_test_loss

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
    lstm_stats_loss = train_eval_lstm(X_tab_train, X_tab_test, X_text_train, X_text_test, y_train, y_test, use_text=False)
    
    print("3. Entraînement du Multimodal LSTM (Stats + Texte sur 150 epochs)...")
    lstm_multi_loss = train_eval_lstm(X_tab_train, X_tab_test, X_text_train, X_text_test, y_train, y_test, use_text=True)
    
    print("\n" + "="*50)
    print("🏆 RÉSULTATS FINAUX (PHASE 6) : LOG-LOSS")
    print("="*50)
    print(f"1. Sklearn Baseline          : {sklearn_loss:.4f}")
    print(f"2. LSTM (Stats-only)         : {lstm_stats_loss:.4f}")
    print(f"3. LSTM (Multimodal)         : {lstm_multi_loss:.4f}")
    print("="*50)
    print("Rappel : Plus le score est proche de 0, meilleures sont les probabilités.")
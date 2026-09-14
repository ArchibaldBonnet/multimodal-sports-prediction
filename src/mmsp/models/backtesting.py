import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from src.mmsp.models.evaluate import get_prepared_data
from src.mmsp.models.lstm import MultimodalNet

def fractional_kelly(prob, odds, fraction=4):
    """
    Calcule la mise selon le Fractional Kelly Criterion.
    Retourne le % du capital à miser (0 si pas de Value Bet).
    """
    if prob * odds <= 1.0:
        return 0.0
    
    b = odds - 1.0
    q = 1.0 - prob
    f_star = (b * prob - q) / b
    
    return max(0.0, f_star / fraction)

def run_financial_backtest():
    print("1. Chargement des données et ré-entraînement du modèle...")
    X_tab_train, X_tab_test, X_text_train, X_text_test, y_train, y_test = get_prepared_data()
    
    df = pd.read_parquet("data/dataset.parquet")
    df_test = df.iloc[int(len(df) * 0.8):].reset_index(drop=True)
    
    # Entraînement rapide du LSTM
    model = MultimodalNet(tabular_dim=12, text_dim=X_text_train.shape[1])
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    for epoch in range(50): 
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tab_train, X_text_train), y_train)
        loss.backward()
        optimizer.step()
        
    print("2. Lancement de la simulation financière sur le set de test...")
    model.eval()
    with torch.no_grad():
        logits = model(X_tab_test, X_text_test)
        probs = torch.softmax(logits, dim=1).numpy()
    
    bankroll = 1000.0
    history = [bankroll]
    paris_places = 0
    
    for idx, row in df_test.iterrows():
        # prob[0]=Away, prob[1]=Draw, prob[2]=Home
        prob_home = probs[idx][2]
        cote_home = row['B365H']
        
        mise_pct = fractional_kelly(prob_home, cote_home, fraction=4)
        mise_eur = bankroll * mise_pct
        
        if mise_eur > 0:
            paris_places += 1
            if row['HomeScore'] > row['AwayScore']:
                bankroll += mise_eur * (cote_home - 1.0)
            else:
                bankroll -= mise_eur
                
        history.append(bankroll)

    # 3. Génération du graphique
    plt.figure(figsize=(10, 5))
    plt.plot(history, color='green' if bankroll >= 1000 else 'red')
    plt.axhline(1000, color='gray', linestyle='--')
    plt.title(f"Bankroll Curve | Paris joués : {paris_places} | Capital : {bankroll:.2f}€")
    plt.xlabel("Matchs simulés")
    plt.ylabel("Capital (€)")
    
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    plt.savefig(reports_dir / "bankroll_curve.png")
    
    print(f"\nPhase 7 terminée ! {paris_places} paris placés. Capital final : {bankroll:.2f}€")
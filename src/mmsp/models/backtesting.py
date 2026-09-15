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
    # Verrou du Value Bet strict
    if prob * odds <= 1.0:
        return 0.0
    
    b = odds - 1.0
    q = 1.0 - prob
    f_star = (b * prob - q) / b
    
    return max(0.0, f_star / fraction)

def run_financial_backtest():
    print("1. Chargement des données et inférence sur le set de test...")
    X_tab_train, X_tab_test, X_text_train, X_text_test, y_train, y_test = get_prepared_data()
    
    df = pd.read_parquet("data/dataset.parquet")
    # Coupure stricte : on ne garde que les 20% récents
    df_test = df.iloc[int(len(df) * 0.8):].reset_index(drop=True)
    
    # Instanciation du modèle avec 12 features tabulaires
    model = MultimodalNet(tabular_dim=12, text_dim=X_text_train.shape[1])
    
    # Entraînement rapide (pour la démo de la Phase 7, tu peux aussi charger des poids sauvegardés)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-3)
    
    for epoch in range(50): 
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tab_train, X_text_train), y_train)
        loss.backward()
        optimizer.step()
        
    print("2. Lancement de la simulation financière sur le set de test (3 issues analysées)...")
    model.eval()
    with torch.no_grad():
        logits = model(X_tab_test, X_text_test)
        probs = torch.softmax(logits, dim=1).numpy()
    
    bankroll = 1000.0
    history = [bankroll]
    bets_log = []
    
    for idx, row in df_test.iterrows():
        # L'IA prédit les 3 probabilités : [Away, Draw, Home]
        probs_match = probs[idx] 
        
        # Cotes du marché
        cotes = [row['B365A'], row['B365D'], row['B365H']] # 0: Away, 1: Draw, 2: Home
        
        # Détermination du résultat réel
        if row['HomeScore'] > row['AwayScore']:
            real_outcome = 2
        elif row['HomeScore'] == row['AwayScore']:
            real_outcome = 1
        else:
            real_outcome = 0

        # Test des 3 issues possibles
        for outcome_idx in range(3):
            p = probs_match[outcome_idx]
            cote = cotes[outcome_idx]
            
            mise_pct = fractional_kelly(p, cote, fraction=4)
            mise_eur = bankroll * mise_pct
            
            if mise_eur > 0:
                won = (outcome_idx == real_outcome)
                profit = mise_eur * (cote - 1.0) if won else -mise_eur
                
                # Traduction de l'issue pour le CSV
                outcome_str = "Away" if outcome_idx == 0 else "Draw" if outcome_idx == 1 else "Home"
                
                bets_log.append({
                    'Date': row['Date'],
                    'Match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                    'Issue_Pari': outcome_str,
                    'Prob_IA': round(p, 3),
                    'Cote_B365': cote,
                    'Prob_Implicite': round(1.0 / cote, 3),
                    'Mise_Euros': round(mise_eur, 2),
                    'Resultat': 'Gagné' if won else 'Perdu',
                    'Profit_Euros': round(profit, 2)
                })
                
                if won:
                    bankroll += profit
                else:
                    bankroll -= mise_eur
                
                # On s'arrête à un pari par match pour éviter le "self-hedging"
                break 
                
        history.append(bankroll)

    # 3. Génération des rapports (CSV et Graphique)
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Sauvegarde du CSV
    df_bets = pd.DataFrame(bets_log)
    csv_path = reports_dir / "matched_bets_log.csv"
    df_bets.to_csv(csv_path, index=False)
    
    # Création du graphique
    plt.figure(figsize=(10, 5))
    plt.plot(history, color='green' if bankroll >= 1000 else 'red')
    plt.axhline(1000, color='gray', linestyle='--')
    plt.title(f"Bankroll Curve | Paris joués : {len(bets_log)} | Capital : {bankroll:.2f}€")
    plt.xlabel("Matchs simulés")
    plt.ylabel("Capital (€)")
    plt.savefig(reports_dir / "bankroll_curve.png")
    
    print(f"\nPhase 7 terminée ! {len(bets_log)} paris placés. Capital final : {bankroll:.2f}€")
    print(f"Journal détaillé sauvegardé dans : {csv_path}")
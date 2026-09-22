import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
from src.mmsp.models.evaluate import get_prepared_data
from src.mmsp.models.lstm import train_multimodal_lstm
import math

def calculate_kelly_with_shrinkage(probs_match, outcome_idx, cote, fraction=4, max_cap=0.05):
    """
    Calcule la mise avec un Kelly Criterion fractionnaire, ajusté par l'entropie (shrinkage)
    et limité par un plafond strict de sécurité (cap).
    """
    p = probs_match[outcome_idx]
    
    # 1. Filtre des cotes extrêmes (Zone de non-calibration)
    # On ignore les cotes > 7.0 (probabilité implicite < 14%)
    if cote > 7.0:
        return 0.0
        
    # Calcul de l'avantage brut
    edge_brut = (p * cote) - 1.0
    
    # Si pas de Value Bet, on ne parie pas
    if edge_brut <= 0.0:
        return 0.0
        
    # 2. Gestion de l'incertitude (Shrinkage basé sur l'entropie)
    # L'entropie maximale pour 3 issues (Victoire, Nul, Défaite) est ln(3)
    entropy = -sum(prob * math.log(prob + 1e-9) for prob in probs_match if prob > 0)
    max_entropy = math.log(3)
    
    # Le shrinkage factor diminue l'edge si le modèle est trop incertain (entropie élevée)
    shrinkage_factor = max(0.0, 1.0 - (entropy / max_entropy))
    edge_ajuste = edge_brut * shrinkage_factor
    
    # 3. Calcul de la mise Kelly avec l'edge ajusté
    b = cote - 1.0
    f_star = edge_ajuste / b
    
    # 4. Application du fractionnement et du Cap maximum
    mise_pct = max(0.0, f_star / fraction)
    mise_pct_limitee = min(mise_pct, max_cap)
    
    return mise_pct_limitee

def simulate_bets(probs, df_test, model_name):
    bankroll = 1000.0
    history = [bankroll]
    bets_log = []
    
    for idx, row in df_test.iterrows():
        probs_match = probs[idx]
        cotes = [row['B365A'], row['B365D'], row['B365H']] 
        
        if row['HomeScore'] > row['AwayScore']:
            real_outcome = 2
        elif row['HomeScore'] == row['AwayScore']:
            real_outcome = 1
        else:
            real_outcome = 0

        for outcome_idx in range(3):
            cote = cotes[outcome_idx]
            p = probs_match[outcome_idx]
            
            # Appel de notre nouvelle fonction Quant
            mise_pct = calculate_kelly_with_shrinkage(probs_match, outcome_idx, cote, fraction=4, max_cap=0.05)
            mise_eur = bankroll * mise_pct
            
            if mise_eur > 0:
                won = (outcome_idx == real_outcome)
                profit = mise_eur * (cote - 1.0) if won else -mise_eur
                
                if won:
                    bankroll += profit
                else:
                    bankroll -= mise_eur
                
                outcome_str = "Away" if outcome_idx == 0 else "Draw" if outcome_idx == 1 else "Home"
                
                bets_log.append({
                    'Model': model_name,
                    'Date': row['Date'],
                    'Match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                    'Issue_Pari': outcome_str,
                    'Prob_IA': round(p, 3),
                    'Prob_B365': round(1.0 / cote, 3),
                    'Cote': cote,
                    'Mise': round(mise_eur, 2),
                    'Profit': round(profit, 2)
                })
                break 
                
        history.append(bankroll)
    return history, bets_log, bankroll

def plot_calibration_and_brier(y_true, probs_dict, reports_dir="reports"):
    """
    Calcule le Brier Score multi-classes et trace les Reliability Diagrams.
    
    y_true : array des vrais résultats (0, 1, 2)
    probs_dict : dictionnaire { "Nom du modèle": probabilités_array }
    """
    Path(reports_dir).mkdir(exist_ok=True)
    
    # Binarisation des labels pour le multi-classes (0, 1, 2)
    classes = np.array([0, 1, 2])
    Y_onehot = label_binarize(y_true, classes=classes)
    
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    
    ax1.plot([0, 1], [0, 1], "k:", label="Parfaite calibration")
    
    print("\n==================================================")
    print("📊 ANALYSE DE CALIBRATION (BRIER SCORE)")
    print("==================================================")
    print("Rappel : Plus le score est proche de 0, mieux le modèle est calibré.\n")
    
    for name, probs in probs_dict.items():
        # 1. Calcul du Brier Score multi-classes (MSE entre probas et one-hot)
        brier_score = np.mean(np.sum((probs - Y_onehot)**2, axis=1))
        print(f"{name:<25} : {brier_score:.4f}")
        
        # 2. Préparation des données pour le diagramme (aplatissement global)
        probs_flat = probs.flatten()
        y_true_flat = Y_onehot.flatten()
        
        # 3. Calcul de la courbe de calibration
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_flat, probs_flat, n_bins=10, strategy='uniform'
        )
        
        # Tracé de la courbe de fiabilité
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{name} (Brier: {brier_score:.3f})")
        
        # Tracé de la distribution des probabilités (Histogramme)
        ax2.hist(probs_flat, range=(0, 1), bins=10, label=name, histtype="step", lw=2)

    ax1.set_ylabel("Fraction réelle de victoires")
    ax1.set_title("Reliability Diagram (Calibration globale)")
    ax1.legend(loc="best")
    
    ax2.set_xlabel("Probabilité prédite par l'IA")
    ax2.set_ylabel("Nombre de prédictions")
    ax2.legend(loc="upper center", ncol=3)
    
    plt.tight_layout()
    save_path = f"{reports_dir}/reliability_diagram.png"
    plt.savefig(save_path)
    print(f"\nGraphique de calibration sauvegardé dans : {save_path}")

def run_financial_backtest():
    """Phase 7 : Génération de la courbe de Bankroll comparative."""
    print("1. Chargement des données et séparation temporelle...")
    X_tab_train, X_tab_test, X_text_train, X_text_test, y_train, y_test = get_prepared_data()
    
    # Chargement complet pour récupérer les cotes (Odds) nécessaires au backtest
    df = pd.read_parquet("data/dataset.parquet")
    df_test = df.iloc[int(len(df) * 0.8):].reset_index(drop=True)
    
    # --- A. Inférence Sklearn Baseline ---
    print("\n2. Inférence Baseline (Sklearn)...")
    baseline_model = joblib.load("models/baseline.pkl")
    
    if X_tab_test.ndim == 3:
        X_tab_test_2d = X_tab_test[:, -1, :] # On prend toutes les colonnes
    else:
        X_tab_test_2d = X_tab_test
        
    probs_baseline = baseline_model.predict_proba(X_tab_test_2d)
    hist_base, log_base, bankroll_base = simulate_bets(probs_baseline, df_test, "Sklearn")
    
    # --- B. Entraînement et Inférence LSTM (Stats-only) ---
    print("\n3. Entraînement et Inférence LSTM (Stats-only)...")
    lstm_stats_model = train_multimodal_lstm(
        X_tab_train, X_tab_test, X_text_train, X_text_test, y_train, y_test, use_text=False
    )
    lstm_stats_model.eval()
    with torch.no_grad():
        dummy_text_test = torch.zeros_like(X_text_test)
        logits_stats = lstm_stats_model(X_tab_test, dummy_text_test)
        probs_lstm_stats = torch.softmax(logits_stats, dim=1).numpy()
        
    hist_stats, log_stats, bankroll_stats = simulate_bets(probs_lstm_stats, df_test, "LSTM Stats-only")

    # --- C. Entraînement et Inférence LSTM Multimodal ---
    print("\n4. Entraînement et Inférence LSTM Multimodal...")
    lstm_multi_model = train_multimodal_lstm(
        X_tab_train, X_tab_test, X_text_train, X_text_test, y_train, y_test, use_text=True
    )
    lstm_multi_model.eval()
    with torch.no_grad():
        logits_multi = lstm_multi_model(X_tab_test, X_text_test)
        probs_lstm_multi = torch.softmax(logits_multi, dim=1).numpy()

    # Rassemblement des probabilités des 3 modèles
    models_probs = {
        "Sklearn Baseline": probs_baseline,
        "LSTM Stats-only": probs_lstm_stats,
        "Multimodal LSTM": probs_lstm_multi
    }
    
    # Génération des scores et du graphique
    plot_calibration_and_brier(y_test, models_probs)    

    hist_multi, log_multi, bankroll_multi = simulate_bets(probs_lstm_multi, df_test, "LSTM Multimodal")

    # --- D. Génération du Graphique Comparatif ---
    print("\n5. Génération des rapports (Phase 7)...")
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Fusion et sauvegarde des logs
    df_all_bets = pd.DataFrame(log_base + log_stats + log_multi)
    df_all_bets.to_csv(reports_dir / "matched_bets_log.csv", index=False)
    
    # Tracé des courbes de bankroll
    plt.figure(figsize=(10, 5))
    plt.plot(hist_base, label=f'Sklearn Baseline ({bankroll_base:.2f}€)', color='blue', alpha=0.7)
    plt.plot(hist_stats, label=f'LSTM Stats-only ({bankroll_stats:.2f}€)', color='green', alpha=0.8)
    plt.plot(hist_multi, label=f'Multimodal LSTM ({bankroll_multi:.2f}€)', color='orange', linewidth=2)
    plt.axhline(1000, color='gray', linestyle='--')
    
    plt.title("Comparaison des Bankrolls : Phase 7")
    plt.xlabel("Matchs simulés")
    plt.ylabel("Capital (€)")
    plt.legend()
    plt.savefig(reports_dir / "bankroll_curve.png")
    
    print("="*60)
    print("🏆 RÉSULTATS FINAUX (SIMULATION FINANCIÈRE)")
    print("="*60)
    print(f"1. Sklearn Baseline          : {bankroll_base:.2f}€")
    print(f"2. LSTM (Stats-only)         : {bankroll_stats:.2f}€")
    print(f"3. LSTM (Multimodal)         : {bankroll_multi:.2f}€")
    print("="*60)
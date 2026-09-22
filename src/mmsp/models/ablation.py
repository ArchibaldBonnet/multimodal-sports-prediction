import pandas as pd
import numpy as np
from pathlib import Path

def run_embedding_ablation(file_path="reports/matched_bets_log.csv"):
    if not Path(file_path).exists():
        print(f"Fichier introuvable : {file_path}")
        return

    # 1. Chargement et détection dynamique des noms
    df = pd.read_csv(file_path)
    model_names = df['Model'].unique()
    
    multi_name = next((n for n in model_names if 'Multi' in n), None)
    stats_name = next((n for n in model_names if 'Stats' in n), None)
    
    if not multi_name or not stats_name:
        print(f"Erreur : Modèles non trouvés dans le CSV. Modèles disponibles : {model_names}")
        return

    # 2. Isolement des modèles
    df_stats = df[df['Model'] == stats_name][['Date', 'Match', 'Issue_Pari', 'Prob_IA', 'Profit']]
    df_multi = df[df['Model'] == multi_name][['Date', 'Match', 'Issue_Pari', 'Prob_IA', 'Profit']]
    
    # 3. Jointure EXTERNE (On garde les matchs où un seul des deux a parié)
    merged = pd.merge(
        df_multi, 
        df_stats, 
        on=['Date', 'Match'], 
        how='outer',
        suffixes=('_multi', '_stats')
    )
    
    # 4. Remplissage des paris non placés
    # Si un modèle n'a pas parié, son profit est de 0. 
    # Pour la proba manquante, on simule une proba à 0 pour maximiser le Delta sur les désaccords.
    merged['Profit_multi'] = merged['Profit_multi'].fillna(0)
    merged['Profit_stats'] = merged['Profit_stats'].fillna(0)
    merged['Prob_IA_multi'] = merged['Prob_IA_multi'].fillna(0)
    merged['Prob_IA_stats'] = merged['Prob_IA_stats'].fillna(0)
    
    # 5. Calcul de la divergence (Delta)
    merged['Delta_Prob'] = abs(merged['Prob_IA_multi'] - merged['Prob_IA_stats'])
    merged_sorted = merged.sort_values(by='Delta_Prob', ascending=False)
    
    top_5_pct_count = max(1, int(len(merged_sorted) * 0.05))
    top_divergences = merged_sorted.head(top_5_pct_count)
    
    # 6. Analyse de l'impact
    multi_wins = len(top_divergences[top_divergences['Profit_multi'] > 0])
    stats_wins = len(top_divergences[top_divergences['Profit_stats'] > 0])
    
    print("==================================================")
    print("🧠 ÉTUDE D'ABLATION : IMPACT DU NLP (TOP 5% DIVERGENCE)")
    print("==================================================")
    print(f"Nombre de matchs analysés (Top 5%) : {top_5_pct_count}")
    print(f"Écart de probabilité moyen sur ces matchs : {top_divergences['Delta_Prob'].mean():.3f}\n")
    
    print("--- PERFORMANCE SUR LES MATCHS À FORTE DIVERGENCE ---")
    print(f"Win Rate (Multimodal) : {multi_wins}/{top_5_pct_count} ({(multi_wins/top_5_pct_count)*100:.1f}%)")
    print(f"Win Rate (Stats-only) : {stats_wins}/{top_5_pct_count} ({(stats_wins/top_5_pct_count)*100:.1f}%)\n")
    
    print(f"Profit cumulé (Multimodal) : {top_divergences['Profit_multi'].sum():.2f} €")
    print(f"Profit cumulé (Stats-only) : {top_divergences['Profit_stats'].sum():.2f} €\n")
    
    print("--- EXEMPLES DE HALLUCINATIONS TEXTUELLES ---")
    hallucinations = top_divergences[top_divergences['Profit_multi'] < 0].head(3)
    for _, row in hallucinations.iterrows():
        pari_m = row['Issue_Pari_multi'] if pd.notna(row['Issue_Pari_multi']) else "Aucun"
        pari_s = row['Issue_Pari_stats'] if pd.notna(row['Issue_Pari_stats']) else "Aucun"
        
        print(f"Match: {row['Match']} ({row['Date']})")
        print(f"  -> Stats-only : Pari={pari_s} | Bilan={row['Profit_stats']}€")
        print(f"  -> Multimodal : Pari={pari_m} (Proba NLP forcée) | Bilan={row['Profit_multi']}€")
        print("-")

if __name__ == "__main__":
    run_embedding_ablation()
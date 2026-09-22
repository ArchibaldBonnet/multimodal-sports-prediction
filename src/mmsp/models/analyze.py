import pandas as pd
from pathlib import Path

def generate_betting_report(file_path="reports/matched_bets_log.csv"):
    if not Path(file_path).exists():
        print(f"Fichier introuvable : {file_path}. Lancez d'abord le backtest.")
        return

    df = pd.read_csv(file_path)

    # 1. Agrégation des statistiques par modèle
    stats = df.groupby('Model').agg(
        Nombre_Paris=('Mise', 'count'),
        Win_Rate_Pct=pd.NamedAgg(column='Profit', aggfunc=lambda x: (x > 0).mean() * 100),
        Mise_Moyenne=('Mise', 'mean'),
        Mise_Max=('Mise', 'max'),
        Cote_Moyenne=('Cote', 'mean'),
        Cote_Max=('Cote', 'max'),
        Profit_Total=('Profit', 'sum')
    ).round(2).reset_index()

    print("==================================================")
    print("📊 STATISTIQUES FINANCIÈRES PAR MODÈLE")
    print("==================================================")
    print(stats.to_markdown(index=False))

    # 2. Détection d'anomalies (Garde-fous)
    print("\n==================================================")
    print("⚠️ DÉTECTION D'ANOMALIES")
    print("==================================================")
    
    # On traque les mises absurdes (ex: > 50€ si le cap est à 5% de 1000€) 
    # ou les cotes extrêmes qui auraient échappé au filtre
    anomalies = df[
        (df['Mise'] <= 0) | 
        (df['Mise'] > 50.0) | 
        (df['Cote'] > 7.0)
    ]
    
    if anomalies.empty:
        print("✅ Aucune anomalie critique détectée dans les logs de paris.")
    else:
        print(f"❌ {len(anomalies)} anomalies trouvées (Mises hors limites ou cotes extrêmes) :")
        print(anomalies[['Date', 'Model', 'Match', 'Mise', 'Cote']].head(10).to_markdown(index=False))

if __name__ == "__main__":
    generate_betting_report()
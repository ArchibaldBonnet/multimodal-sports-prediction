import pandas as pd
import numpy as np
from pathlib import Path
import glob

def build_canonical_dataset():
    csv_files = glob.glob("E0*.csv")
    if not csv_files:
        print("Erreur: Aucun fichier CSV de type E0 trouvé à la racine.")
        return

    dataframes = []
    for file in csv_files:
        print(f"Chargement de {file}...")
        df = pd.read_csv(file, parse_dates=['Date'], dayfirst=True)
        dataframes.append(df)

    full_df = pd.concat(dataframes, ignore_index=True)
    full_df = full_df.dropna(subset=['HomeTeam', 'AwayTeam']) 
    full_df = full_df.sort_values(by='Date').reset_index(drop=True)
    
    full_df = full_df.rename(columns={'FTHG': 'HomeScore', 'FTAG': 'AwayScore'})
    
    # Calcul des points
    full_df['Home_Points'] = np.where(full_df['HomeScore'] > full_df['AwayScore'], 3, 
                             np.where(full_df['HomeScore'] == full_df['AwayScore'], 1, 0))
    full_df['Away_Points'] = np.where(full_df['AwayScore'] > full_df['HomeScore'], 3, 
                             np.where(full_df['HomeScore'] == full_df['AwayScore'], 1, 0))
    
    def rolling_avg(df, team_col, stat_col):
        return df.groupby(team_col)[stat_col].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

    # --- ANCIENNES STATS ---
    full_df['Home_AvgScored_5'] = rolling_avg(full_df, 'HomeTeam', 'HomeScore')
    full_df['Home_AvgConceded_5'] = rolling_avg(full_df, 'HomeTeam', 'AwayScore')
    full_df['Home_Points_5'] = rolling_avg(full_df, 'HomeTeam', 'Home_Points')
    
    full_df['Away_AvgScored_5'] = rolling_avg(full_df, 'AwayTeam', 'AwayScore')
    full_df['Away_AvgConceded_5'] = rolling_avg(full_df, 'AwayTeam', 'HomeScore')
    full_df['Away_Points_5'] = rolling_avg(full_df, 'AwayTeam', 'Away_Points')
    
    # --- NOUVELLES STATS AVANCÉES (Tirs, Cadrés, Corners) ---
    full_df['Home_Shots_5'] = rolling_avg(full_df, 'HomeTeam', 'HS')
    full_df['Home_ShotsTarget_5'] = rolling_avg(full_df, 'HomeTeam', 'HST')
    full_df['Home_Corners_5'] = rolling_avg(full_df, 'HomeTeam', 'HC')
    
    full_df['Away_Shots_5'] = rolling_avg(full_df, 'AwayTeam', 'AS')
    full_df['Away_ShotsTarget_5'] = rolling_avg(full_df, 'AwayTeam', 'AST')
    full_df['Away_Corners_5'] = rolling_avg(full_df, 'AwayTeam', 'AC')

    # Liste des 18 variables désormais requises
    required_cols = [
        'Date', 'HomeTeam', 'AwayTeam', 'HomeScore', 'AwayScore', 'B365H', 'B365D', 'B365A', 
        'Home_AvgScored_5', 'Home_AvgConceded_5', 'Home_Points_5', 'Away_AvgScored_5', 'Away_AvgConceded_5', 'Away_Points_5',
        'Home_Shots_5', 'Home_ShotsTarget_5', 'Home_Corners_5', 'Away_Shots_5', 'Away_ShotsTarget_5', 'Away_Corners_5'
    ]
    
    # Nettoyage des matchs incomplets (ex: les premiers matchs de la saison sans historique)
    full_df = full_df.dropna(subset=required_cols)

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    full_df.to_parquet(out_dir / "dataset.parquet", index=False)
    
    print(f"\nBase de données reconstruite avec succès ! ({len(full_df)} matchs conservés avec stats avancées)")
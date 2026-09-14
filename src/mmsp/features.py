import pandas as pd
from pathlib import Path

def calculate_rolling_stats(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Calcule les statistiques glissantes pré-match pour chaque équipe.
    """
    # 1. Calcul des points rapportés par chaque match
    df['HomePoints'] = (df['HomeScore'] > df['AwayScore']) * 3 + (df['HomeScore'] == df['AwayScore']) * 1
    df['AwayPoints'] = (df['AwayScore'] > df['HomeScore']) * 3 + (df['HomeScore'] == df['AwayScore']) * 1

    # 2. Séparation temporaire pour aligner toutes les performances d'une équipe sur une seule colonne
    home_df = df[['Date', 'HomeTeam', 'HomeScore', 'AwayScore', 'HomePoints']].rename(
        columns={'HomeTeam': 'Team', 'HomeScore': 'Scored', 'AwayScore': 'Conceded', 'HomePoints': 'Points'}
    )
    away_df = df[['Date', 'AwayTeam', 'AwayScore', 'HomeScore', 'AwayPoints']].rename(
        columns={'AwayTeam': 'Team', 'AwayScore': 'Scored', 'HomeScore': 'Conceded', 'AwayPoints': 'Points'}
    )

    # 3. Tri chronologique absolu par équipe
    team_history = pd.concat([home_df, away_df]).sort_values(by=['Team', 'Date'])

    # 4. Calcul décalé (shift) pour éviter les fuites de données du futur
    team_history[f'AvgScored_{window}'] = team_history.groupby('Team')['Scored'].transform(lambda x: x.shift(1).rolling(window).mean())
    team_history[f'AvgConceded_{window}'] = team_history.groupby('Team')['Conceded'].transform(lambda x: x.shift(1).rolling(window).mean())
    team_history[f'Points_{window}'] = team_history.groupby('Team')['Points'].transform(lambda x: x.shift(1).rolling(window).sum())

    return team_history[['Date', 'Team', f'AvgScored_{window}', f'AvgConceded_{window}', f'Points_{window}']]

def build_features() -> pd.DataFrame:
    """
    Charge l'historique canonique, génère les features tabulaires et les fusionne par match.
    """
    input_path = Path("data/matches.parquet")
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} introuvable. Exécutez d'abord la Phase 1.")
        
    df = pd.read_parquet(input_path).sort_values('Date')
    
    # Génération des dynamiques sur 5 matchs
    team_stats = calculate_rolling_stats(df, window=5)
    
    # Jointure pour l'équipe à Domicile
    df = df.merge(
        team_stats, 
        left_on=['Date', 'HomeTeam'], 
        right_on=['Date', 'Team'], 
        how='left'
    ).rename(
        columns={'AvgScored_5': 'Home_AvgScored_5', 'AvgConceded_5': 'Home_AvgConceded_5', 'Points_5': 'Home_Points_5'}
    ).drop(columns=['Team'])
    
    # Jointure pour l'équipe à l'Extérieur
    df = df.merge(
        team_stats, 
        left_on=['Date', 'AwayTeam'], 
        right_on=['Date', 'Team'], 
        how='left'
    ).rename(
        columns={'AvgScored_5': 'Away_AvgScored_5', 'AvgConceded_5': 'Away_AvgConceded_5', 'Points_5': 'Away_Points_5'}
    ).drop(columns=['Team'])

    output_path = Path("data/features.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Phase 2 réussie : Features sauvegardées dans {output_path}")
    
    return df
import pandas as pd
from pathlib import Path

def build_dataset() -> pd.DataFrame:
    """
    Phase 3: Finalise le dataset canonique en supprimant les NaNs 
    (issus des moyennes glissantes du début de saison).
    """
    input_path = Path("data/features.parquet")
    if not input_path.exists():
        raise FileNotFoundError("features.parquet introuvable. La Phase 2 a-t-elle échoué ?")
        
    df = pd.read_parquet(input_path)
    
    # Règle absolue de la Phase 3 : "no silent NaNs"
    # On supprime toutes les lignes incomplètes (les 5 premiers matchs de chaque équipe)
    df_clean = df.dropna().copy()
    
    # Tri temporel strict pour respecter le "temporal split" exigé par la configuration
    df_clean = df_clean.sort_values('Date').reset_index(drop=True)
    
    # Vérification de sécurité interne
    assert df_clean.isna().sum().sum() == 0, "Erreur fatale : Il reste des NaNs silencieux !"
    assert df_clean['match_id'].is_unique, "Erreur fatale : Plusieurs lignes pour un même match !"

    output_path = Path("data/dataset.parquet")
    df_clean.to_parquet(output_path, index=False)
    
    print(f"Phase 3 réussie : Dataset finalisé avec {len(df_clean)} matchs 100% complets dans {output_path}")
    
    return df_clean
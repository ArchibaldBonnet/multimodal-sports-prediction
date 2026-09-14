import pandas as pd
from pathlib import Path

def test_matches_parquet_exists_and_unique():
    file_path = Path("data/matches.parquet")
    
    assert file_path.exists(), "Le fichier matches.parquet n'existe pas."
    
    df = pd.read_parquet(file_path)
    assert not df.empty, "Le dataframe est vide."
    
    # Vérifie qu'il n'y a aucun doublon
    assert df['match_id'].is_unique, "Doublons détectés dans les match_id"
    
    # Vérifie l'absence de valeurs nulles sur la clé
    assert df['match_id'].isna().sum() == 0, "Des match_id sont NaN"

def test_dataset_parquet_ready_for_ml():
    """Valide les exigences de la Phase 3"""
    file_path = Path("data/dataset.parquet")
    assert file_path.exists(), "Le fichier final dataset.parquet n'a pas été généré."
    
    df = pd.read_parquet(file_path)
    
    # Contrainte 1 : Une ligne par match (unicité du match_id)
    assert df['match_id'].is_unique, "Le dataset contient plusieurs lignes pour un même match."
    
    # Contrainte 2 : No silent NaNs
    assert df.isna().sum().sum() == 0, "Le dataset contient des valeurs manquantes, ce qui causera un crash du modèle Sklearn."
    
    # Vérification de l'ordre temporel (crucial pour le split)
    assert df['Date'].is_monotonic_increasing, "Le dataset n'est pas strictement trié par date."
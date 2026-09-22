import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss, accuracy_score

def create_target(df: pd.DataFrame) -> pd.Series:
    """
    Convertit les scores en résultat final : 0 (Victoire Extérieur), 1 (Nul), 2 (Victoire Domicile).
    C'est le format standard pour la classification multi-classes.
    """
    conditions = [
        df['HomeScore'] < df['AwayScore'],
        df['HomeScore'] == df['AwayScore'],
        df['HomeScore'] > df['AwayScore']
    ]
    choices = [0, 1, 2]
    return pd.Series(np.select(conditions, choices, default=np.nan))

def train_sklearn_baseline():
    """
    Entraîne la baseline Sklearn sur une séparation temporelle stricte
    et évalue les probabilités via le log-loss.
    """
    dataset_path = Path("data/dataset.parquet")
    if not dataset_path.exists():
        raise FileNotFoundError("dataset.parquet introuvable. Lancez la Phase 3.")
        
    df = pd.read_parquet(dataset_path)
    
    # 1. Définition de la cible et des variables explicatives (features)
    y = create_target(df)
    features = [
    'Home_AvgScored_5', 'Home_AvgConceded_5', 'Home_Points_5', 
    'Away_AvgScored_5', 'Away_AvgConceded_5', 'Away_Points_5',
    'Home_Shots_5', 'Home_ShotsTarget_5', 'Home_Corners_5', 
    'Away_Shots_5', 'Away_ShotsTarget_5', 'Away_Corners_5'
]
    X = df[features]
    
    # 2. Temporal Split (Séparation Temporelle)
    # Pour respecter la consigne du projet, on ne mélange SURTOUT PAS les données (pas de train_test_split aléatoire).
    # On prend les 80% les plus anciens pour l'entraînement, et les 20% les plus récents pour le test.
    split_index = int(len(df) * 0.8)
    
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print(f"Entraînement sur {len(X_train)} matchs passés...")
    print(f"Test sur {len(X_test)} matchs récents...")
    
    # 3. Création et entraînement du pipeline (Standardisation + Régression Logistique)
    # On limite les itérations et on ajoute un peu de régularisation (C=0.1) pour éviter le surapprentissage
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.1, random_state=42))
    model.fit(X_train, y_train)
    
    # 4. Évaluation (Log-loss et Précision)
    # Le log-loss sanctionne lourdement un modèle qui est "très sûr de lui mais qui se trompe"
    y_pred_proba = model.predict_proba(X_test)
    y_pred_class = model.predict(X_test)
    
    loss = log_loss(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred_class)
    
    print("\n=== RÉSULTATS SKLEARN BASELINE ===")
    print(f"Log-loss (Plus c'est proche de 0, mieux c'est) : {loss:.4f}")
    print(f"Précision (Accuracy) : {acc*100:.2f}%")
    print("==================================")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Sauvegarde du modèle sur le disque
    save_path = models_dir / "baseline.pkl"
    joblib.dump(model, save_path)
    print(f"Modèle sauvegardé dans : {save_path}")
    
    return model
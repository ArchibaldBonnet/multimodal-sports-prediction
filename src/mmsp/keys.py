import re
import pandas as pd

# Petit dictionnaire pour les alias très communs qui ne sont pas résolus par le nettoyage générique.
# Les clés doivent être écrites telles qu'elles ressortent APRES l'étape 3 du nettoyage (sans espaces).
TEAM_MAPPING = {
    "ManUnited": "ManchesterUnited",
    "ManCity": "ManchesterCity",
    "Spurs": "Tottenham",
    "Wolves": "Wolverhampton",
    "NottmForest": "NottinghamForest"
}

def clean_team_name(name: str) -> str:
    """
    Nettoie et standardise le nom d'une équipe de football pour assurer une jointure parfaite.
    """
    if pd.isna(name):
        return ""
        
    name = str(name).strip()
    
    # 1. Suppression des acronymes génériques avec frontières de mots (\b) pour éviter de couper des vrais noms
    name = re.sub(r'\bFC\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\bAFC\b', '', name, flags=re.IGNORECASE)
    
    # 2. Standardisation des abréviations évidentes avant suppression de la ponctuation
    name = name.replace("Utd", "United")
    
    # 3. Suppression stricte de TOUT ce qui n'est pas une lettre (espaces, tirets, apostrophes)
    name = re.sub(r'[^a-zA-Z]', '', name)
    
    # 4. Remplacement final via le dictionnaire (ex: Spurs -> Tottenham)
    return TEAM_MAPPING.get(name, name)

def generate_match_id(date_series: pd.Series, home_series: pd.Series, away_series: pd.Series) -> pd.Series:
    """
    Génère la "canonical match key" unique pour chaque match : YYYY-MM-DD_HomeTeam_AwayTeam.
    """
    # 1. Formatage robuste de la date en YYYY-MM-DD
    # On utilise format="mixed" pour gérer différents formats potentiels selon la source
    dates_formatted = pd.to_datetime(date_series, format="mixed", dayfirst=True).dt.strftime('%Y-%m-%d')
    
    # 2. Nettoyage vectorisé des noms d'équipes (application de clean_team_name à toute la colonne)
    homes_clean = home_series.apply(clean_team_name)
    aways_clean = away_series.apply(clean_team_name)
    
    # 3. Concaténation pour créer la clé canonique
    return dates_formatted + "_" + homes_clean + "_" + aways_clean
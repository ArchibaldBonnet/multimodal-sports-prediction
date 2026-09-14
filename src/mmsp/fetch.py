import os
import time
import requests
import pandas as pd
from pathlib import Path
from src.mmsp.keys import generate_match_id

def fetch_premier_league_data(api_key: str = None) -> pd.DataFrame:
    """
    Télécharge l'historique complet des matchs de Premier League depuis l'API football-data.org.
    """
    api_key = api_key or os.getenv("FOOTBALL_DATA_API_KEY")
    if not api_key:
        raise ValueError("Clé API manquante. Définissez FOOTBALL_DATA_API_KEY.")

    headers = {"X-Auth-Token": api_key}
    all_parsed_matches = []
    
    # Boucle sur les 7 dernières saisons (de 2018/2019 à la saison actuelle)
    seasons = range(2018, 2027) 
    
    print("Interrogation de l'API football-data.org pour l'historique...")
    for season in seasons:
        print(f"-> Récupération de la saison {season}/{season+1}...")
        url = f"https://api.football-data.org/v4/competitions/PL/matches?season={season}"
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            matches = data.get('matches', [])
            
            for match in matches:
                # On ne conserve que les matchs terminés
                if match["status"] == "FINISHED":
                    all_parsed_matches.append({
                        "Date": match["utcDate"],
                        "HomeTeam": match["homeTeam"]["name"],
                        "AwayTeam": match["awayTeam"]["name"],
                        "HomeScore": match["score"]["fullTime"]["home"],
                        "AwayScore": match["score"]["fullTime"]["away"]
                    })
        else:
            print(f"Erreur {response.status_code} sur la saison {season}: {response.text}")
            
        # L'API gratuite autorise 10 requêtes par minute. On temporise 6 secondes entre chaque appel.
        time.sleep(6)

    df = pd.DataFrame(all_parsed_matches)
    
    # Création de la clé canonique
    df['match_id'] = generate_match_id(df['Date'], df['HomeTeam'], df['AwayTeam'])

    # Sauvegarde au format Parquet
    output_path = Path("data/matches.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f"\nSuccès : {len(df)} matchs historiques sauvegardés dans {output_path}")
    return df
import os
import time
import requests
import torch
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

nlp_model = SentenceTransformer('all-MiniLM-L6-v2')

def fetch_guardian_news(team_name: str, match_date: str) -> str:
    api_key = os.getenv("GUARDIAN_API_KEY")
    if not api_key:
        raise ValueError("La variable d'environnement GUARDIAN_API_KEY est manquante.")

    url = "https://content.guardianapis.com/search"
    params = {
        "q": f'"{team_name}"',
        "section": "football",
        "to-date": match_date,
        "show-fields": "bodyText",
        "page-size": 2,
        "api-key": api_key
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            articles = response.json().get('response', {}).get('results', [])
            if articles:
                return " ".join([a.get('fields', {}).get('bodyText', '') for a in articles])
    except Exception as e:
        print(f"Erreur requête Guardian pour {team_name}: {e}")
        
    time.sleep(0.08) # Limite de requêtes Guardian
    return f"{team_name} match preview."

def get_text_features(df: pd.DataFrame) -> torch.Tensor:
    """Génère les embeddings réels (384 + 384 = 768 dimensions) et les stocke en cache."""
    cache_path = Path("data/guardian_embeddings.pt")
    
    if cache_path.exists():
        print("Chargement des embeddings Guardian depuis le cache disque...")
        return torch.load(cache_path, weights_only=True)
        
    print("Extraction des articles Guardian et calcul des embeddings NLP...")
    all_embeddings = []
    
    for idx, row in df.iterrows():
        date_str = str(row['Date'])[:10]
        home, away = row['HomeTeam'], row['AwayTeam']
        
        home_text = fetch_guardian_news(home, date_str)
        away_text = fetch_guardian_news(away, date_str)
        
        # Type 1 : Contexte Domicile (384 dim)
        emb_type_1 = torch.tensor(nlp_model.encode(home_text), dtype=torch.float32)
        # Type 2 : Contexte Extérieur (384 dim)
        emb_type_2 = torch.tensor(nlp_model.encode(away_text), dtype=torch.float32)
        
        all_embeddings.append(torch.cat([emb_type_1, emb_type_2]))
        
        if (idx + 1) % 25 == 0 or (idx + 1) == len(df):
            print(f"Progression NLP : {idx + 1}/{len(df)} matchs encodés...")
            
    tensor_matrix = torch.stack(all_embeddings)
    torch.save(tensor_matrix, cache_path)
    return tensor_matrix
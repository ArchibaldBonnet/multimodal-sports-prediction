# Multimodal Sports Prediction

Calibrated sklearn baseline vs. a multimodal LSTM (tabular stats + text embeddings) for Premier League match outcome prediction, validated economically via value betting and fractional Kelly criterion.

Ce projet évalue rigoureusement l'impact prédictif et financier de l'intégration du traitement du langage naturel (NLP) dans la modélisation de séries temporelles sportives. Il démontre la construction d'un pipeline complet : de l'ingestion de données hétérogènes à l'évaluation probabiliste (Brier Score) et la simulation financière sous contraintes de risque quantitatif.

## 📊 Principaux Résultats & Insights (Phase 7 & 8)

L'évaluation ne se limite pas à la *log-loss* ou à l'*accuracy*, mais simule une stratégie de pari de valeur (Value Betting) utilisant le Kelly Criterion Fractionnaire sur un capital initial de 1000€.

| Modèle | Capital Final | ROI | Comportement & Calibration | 
 | ----- | ----- | ----- | ----- | 
| **LSTM (Stats-only)** | **1351.34 €** | **+35.1%** | Prédictions conservatrices, excellente gestion par la stratégie de Kelly. | 
| **Sklearn Baseline** | 833.12 € | \-16.6% | Calibration correcte (Brier: 0.644) mais manque d'avantage prédictif (edge). | 
| **LSTM (Multimodal)** | 517.28 € | \-48.2% | Haute volatilité due au bruit textuel (sur-confiance probabiliste). | 

### L'Insight Scientifique : Le paradoxe du NLP (Ablation Study)

Le modèle Multimodal détruit le capital à cause d'une **sur-confiance systémique** prouvée par son diagramme de fiabilité (Brier: 0.692). Les articles de presse génèrent des certitudes extrêmes face auxquelles le Kelly Criterion réagit en sur-misant.

Cependant, une étude d'ablation isolant le top 5% des matchs présentant la plus forte divergence entre le modèle Stats et le modèle Multimodal a révélé un phénomène inattendu :

* **Win Rate du NLP sur ces anomalies :** 77.3% (17 succès sur 22 matchs).

* **Conclusion :** Le NLP est un parieur quotidien désastreux à cause du bruit médiatique, mais un **excellent détecteur de cygnes noirs** (blessures clés de dernière minute, crises internes non reflétées par les statistiques glissantes).

### Next Steps : Architecture "Mixture of Experts" (MoE)

Pour une mise en production, l'architecture optimale n'est pas une simple fusion des vecteurs, mais un réseau hybride (Gating Network) : le modèle *Stats-only* pilote la stratégie quotidienne, tandis que le modèle *NLP* s'active uniquement comme détecteur de signaux asymétriques pour capturer les anomalies statistiques.

## 🏗️ Architecture & Pipeline de Données

Les deux modèles (Sklearn et LSTM) consomment rigoureusement le même dataset canonique et respectent la même séparation temporelle via `config.yaml`. Cette symétrie garantit l'intégrité de l'évaluation.

1. **Ingestion Hétérogène :** 12 caractéristiques tabulaires (buts, points, tirs, corners) fusionnées avec des historiques d'articles récupérés via l'API The Guardian.

2. **Feature Engineering :** Conversion du NLP en embeddings mathématiques via `sentence-transformers`.

3. **Risk Management :** L'algorithme de staking intègre un plafonnement du capital (Cap de 5%) et un filtre quantitatif sur l'entropie (Shrinkage) pour pénaliser les prédictions incertaines et ignorer la longue traîne (cotes > 7.0).

## 🚀 Installation & Reproductibilité

Le projet requiert un environnement virtuel propre (`.venv`). Les dépendances incluent `pandas`, `pyarrow`, `scikit-learn`, `torch`, `sentence-transformers`, `matplotlib`, `requests` et `pytest`.

**1. Cloner et installer le package en mode éditable**

```
pip install -e .

```

**2. Installer les dépendances du pipeline complet**

```
pip install -r requirements.txt

```

## 🛠️ Utilisation (Exécution des Phases)

Le pipeline complet est pilotable via le script unifié `main.py`.

```
python main.py --help
python main.py fetch              # Phase 1 : Collecte des données
python main.py build-features     # Phase 2/3 : Création du dataset canonique
python main.py train-baseline     # Phase 4 : Modèle Sklearn 
python main.py train-lstm         # Phase 5 : Entraînement Deep Learning
python main.py evaluate           # Phase 6 : Calibration probabiliste
python main.py backtest           # Phase 7 : Simulation financière (Bankroll)
python main.py ablation           # Bonus phase : NLP Impact
python main.py analyze            # Bonus Phase : Analyse bets 3 modèles

```

## 📋 Statut du Projet

| Phase | Livrable | Statut | 
 | ----- | ----- | ----- | 
| 0 | Squelette : installation, `main.py --help` fonctionnel | ✅ | 
| 1 | `matches.parquet` avec `match_id` unique, tests validés | ✅ | 
| 2 | Features tabulaires jointes par `match_id` | ✅ | 
| 3 | `dataset.parquet` finalisé (sans NaNs silencieux) | ✅ | 
| 4 | Baseline Sklearn entraînée sur la séparation temporelle | ✅ | 
| 5 | LSTM Multimodal codé avec courbes de pertes | ✅ | 
| 6 | Comparaison probabiliste (Brier Score & Reliability) | ✅ | 
| 7 | Simulation de bankroll via `matched_bets_log.csv` | ✅ | 
| 8 | README complet & Étude d'Ablation documentée | ✅ | 

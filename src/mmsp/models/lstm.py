import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from src.mmsp.nlp import get_text_features

# --- ARCHITECTURE DU RÉSEAU MULTIMODAL ---
class MultimodalNet(nn.Module):
    def __init__(self, tabular_dim, text_dim, hidden_dim=24, num_classes=3):
        super(MultimodalNet, self).__init__()
        
        # Le LSTM gère la séquence tabulaire
        self.lstm = nn.LSTM(tabular_dim, hidden_dim, batch_first=True)
        
        # Le réseau dense gère le texte avec un fort Dropout
        self.text_fc = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4) # Désactive 40% des neurones aléatoirement
        )
        
        # La fusion ajoute aussi du Dropout avant la classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4), # Désactive 40% des neurones aléatoirement
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, tabular_x, text_x):
        lstm_out, (hn, cn) = self.lstm(tabular_x)
        tab_features = hn[-1] 
        txt_features = self.text_fc(text_x)
        combined = torch.cat((tab_features, txt_features), dim=1)
        return self.classifier(combined)

# --- SCRIPT D'ENTRAÎNEMENT ---
def train_multimodal_lstm():
    dataset_path = Path("data/dataset.parquet")
    df = pd.read_parquet(dataset_path)
    
    # Préparation Cible
    conditions = [df['HomeScore'] < df['AwayScore'], df['HomeScore'] == df['AwayScore'], df['HomeScore'] > df['AwayScore']]
    y = torch.tensor(np.select(conditions, [0, 1, 2], default=np.nan), dtype=torch.long)
    
    # Préparation Statistiques
    features = ['Home_AvgScored_5', 'Home_AvgConceded_5', 'Home_Points_5', 'Away_AvgScored_5', 'Away_AvgConceded_5', 'Away_Points_5']
    X_tab = StandardScaler().fit_transform(df[features])
    X_tab = torch.tensor(X_tab, dtype=torch.float32).unsqueeze(1) # Ajout de la dimension temps pour le LSTM
    
    # Préparation Texte
    X_text = get_text_features(df)

    
    # Séparation temporelle stricte (80/20)
    split_idx = int(len(df) * 0.8)
    X_tab_train, X_tab_test = X_tab[:split_idx], X_tab[split_idx:]
    X_text_train, X_text_test = X_text[:split_idx], X_text[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Initialisation
    model = MultimodalNet(tabular_dim=len(features), text_dim=X_text.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    train_losses, test_losses = [], []
    print("Entraînement du Multimodal LSTM...")
    
    for epoch in range(150):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tab_train, X_text_train), y_train)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(X_tab_test, X_text_test), y_test)
            
        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
        
        if (epoch + 1) % 30 == 0:
            print(f"Epoch {epoch+1}/150 | Test Log-Loss: {test_loss.item():.4f}")

    # Export des courbes
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title("Multimodal LSTM Loss")
    plt.legend()
    Path("reports").mkdir(exist_ok=True)
    plt.savefig("reports/lstm_loss_curves.png")
    print("\nCourbes sauvegardées dans reports/lstm_loss_curves.png")
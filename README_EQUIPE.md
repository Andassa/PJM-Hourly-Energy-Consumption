# Guide équipe — projet PJM (lancement, fichiers, résultats)

Ce document est le **manuel opérationnel** du dépôt. Il complète le [README principal](README.md) (contexte théorique, maths, architecture détaillée).

**Dépôt GitHub :** [PJM-Hourly-Energy-Consumption](https://github.com/Andassa/PJM-Hourly-Energy-Consumption)

**Public visé :** tous les membres du groupe — pour cloner, installer, lancer les scripts, comprendre **chaque sortie** et interpréter les figures pour le rapport / la soutenance.

---

## Table des matières

1. [Objectif et question du projet](#1-objectif-et-question-du-projet)
2. [Roadmap globale (de zéro au livrable)](#2-roadmap-globale-de-zéro-au-livrable)
3. [Arborescence et rôle de chaque fichier](#3-arborescence-et-rôle-de-chaque-fichier)
4. [Après `git pull` : ce qui est sur GitHub ou non](#4-après-git-pull--ce-qui-est-sur-github-ou-non)
5. [Installation](#5-installation)
6. [Parcours A — Vérifier le projet (sans réentraîner)](#6-parcours-a--vérifier-le-projet-sans-réentraîner)
7. [Parcours B — Pipeline complet depuis zéro](#7-parcours-b--pipeline-complet-depuis-zéro)
8. [Parcours C — Entraîner et évaluer les 3 modèles](#8-parcours-c--entraîner-et-évaluer-les-3-modèles)
9. [Référence des commandes (entrée → sortie)](#9-référence-des-commandes-entrée--sortie)
10. [Fichier `config.yaml`](#10-fichier-configyaml)
11. [Roadmap d’interprétation des résultats](#11-roadmap-dinterprétation-des-résultats)
12. [Catalogue des fichiers de résultats](#12-catalogue-des-fichiers-de-résultats)
13. [Dépannage](#13-dépannage)
14. [Liens utiles](#14-liens-utiles)

---

## 1. Objectif et question du projet

### 1.1 Contexte

La zone **PJME** (PJM East) est une partie du réseau électrique américain. Le fichier `PJME_hourly.csv` contient la **charge horaire** observée (`PJME_MW`, en mégawatts).

### 1.2 Question de recherche

> Étant donnés les **168 dernières heures** de charge et de variables calendaires, peut-on prédire les **24 prochaines heures** de consommation ? Quelle architecture RNA (LSTM, Bi-LSTM, LSTM + attention) performe le mieux ?

### 1.3 Contraintes méthodologiques (à connaître pour la soutenance)

| Règle | Pourquoi |
|--------|----------|
| Découpage **chronologique** 80 % / 10 % / 10 % | Ne pas mélanger le futur dans le passé (pas de fuite temporelle) |
| **Min-Max** calculé sur le **train** seulement | Même scaler appliqué au val/test sans tricher |
| Fenêtre **168 h → 24 h** | Contexte d’une semaine pour prévoir la journée suivante |
| Métriques finales en **MW** (dénormalisées) | Comparaison lisible avec la charge réelle |

### 1.4 Livrables attendus

| Livrable | Fichier / emplacement |
|----------|------------------------|
| Code reproductible | `energy_forecast/src/` |
| 3 modèles entraînés | `results/checkpoints/best_*.pth` |
| Tableau de métriques | `results/metrics.csv` |
| Figures | `results/plots/` |
| Rapport lisible | `results/report.html` |
| Comparaison statistique | `results/dm_results.csv` |
| Présentation / PDF | À produire par l’équipe (export navigateur ou slides) |

---

## 2. Roadmap globale (de zéro au livrable)

```mermaid
flowchart LR
  A[Données brutes CSV] --> B[Notebook EDA]
  B --> C[CSV nettoyé]
  C --> D[Features + split + fenêtres]
  D --> E[train.pkl / val.pkl / test.pkl]
  E --> F[train.py x3 modèles]
  F --> G[best_*.pth]
  G --> H[evaluate.py]
  H --> I[metrics.csv + plots]
  G --> J[DM + heatmaps + report.html]
  I --> J
```

### Ordre des étapes (numéroté)

| # | Action | Script / outil | Sortie principale |
|---|--------|----------------|-------------------|
| 0 | Installer Python, Git, dépendances | `pip install -r requirements.txt` | environnement prêt |
| 1 | Copier / vérifier les données | `data/raw/PJME_hourly.csv` | fichier brut |
| 2 | Explorer et nettoyer (DST, trous) | notebook `01_eda_pjm_hourly_load.ipynb` | `PJME_hourly_clean.csv` |
| 3 | Features calendaires | `preprocessing.py` | `PJME_hourly_with_features.csv` |
| 4 | Split + Min-Max + fenêtres | `build_splits_sequences.py` ou `preprocessing.py --all` | `train/val/test.pkl`, scaler, meta |
| 5 | Entraîner LSTM, Bi-LSTM, attention | `train.py` (×3, `config.yaml`) | `best_*.pth`, `history_*.json` |
| 6 | Évaluer sur le test | `evaluate.py` (×3 ou `--checkpoint`) | terminal JSON + `metrics.csv` + PNG |
| 7 | Courbes de loss | `plot_training_history.py` | `loss_*.png` |
| 8 | Test Diebold–Mariano | `diebold_mariano.py` | `dm_results.csv` |
| 9 | Heatmaps attention | `plot_attention_heatmap.py` | `attn_heatmap_*.png` |
| 10 | Rapport HTML | `generate_report.py` | `report.html` |

**Après un simple `git pull`**, les étapes **0, 1, 2, 3, 5, 6, 7, 8, 9, 10** sont déjà faites sur le dépôt distant : il reste surtout **l’étape 4** (recréer les `.pkl` locaux) pour pouvoir relancer train/evaluate.

---

## 3. Arborescence et rôle de chaque fichier

Tous les chemins sont relatifs à la **racine du dépôt Git** (dossier qui contient `requirements.txt`).

```
PJM Hourly Energy Consumption/
├── requirements.txt          # Dépendances Python (PyTorch, pandas, etc.)
├── README.md                 # Documentation académique complète
├── README_EQUIPE.md          # Ce guide
│
└── energy_forecast/
    ├── config.yaml           # Chemins, hyperparamètres, modèle actif
    │
    ├── data/
    │   ├── raw/
    │   │   └── PJME_hourly.csv              # Données Kaggle brutes
    │   └── processed/
    │       ├── PJME_hourly_clean.csv          # Après EDA (grille 1 h, DST)
    │       ├── PJME_hourly_with_features.csv  # + hour, dow, month, fériés…
    │       ├── train.pkl / val.pkl / test.pkl # Fenêtres (NON sur GitHub)
    │       ├── sequence_scaler.joblib       # MinMaxScaler + noms colonnes
    │       └── sequence_meta.json           # Dates split, shapes, features
    │
    ├── notebooks/
    │   └── 01_eda_pjm_hourly_load.ipynb     # EDA + export clean.csv
    │
    ├── src/
    │   ├── preprocessing.py                 # Features (+ --all → splits)
    │   ├── feature_engineering.py           # hour, dow, is_holiday…
    │   ├── build_splits_sequences.py        # Split 80/10/10 + .pkl
    │   ├── models.py                        # LSTM, Bi-LSTM, LSTM+attention
    │   ├── train.py                         # Entraînement + early stopping
    │   ├── evaluate.py                      # Test MW + métriques + figures
    │   ├── eval_utils.py                    # Utilitaires partagés (DM, heatmap)
    │   ├── plot_training_history.py         # Courbes train/val loss
    │   ├── diebold_mariano.py               # Test statistique entre modèles
    │   ├── plot_attention_heatmap.py        # Poids d’attention Bahdanau
    │   └── generate_report.py               # Assemble report.html
    │
    └── results/
        ├── metrics.csv                      # MAE, RMSE, MAPE (test)
        ├── dm_results.csv                   # Diebold–Mariano
        ├── report.html                      # Rapport final (navigateur)
        ├── checkpoints/
        │   ├── best_lstm.pth
        │   ├── best_bilstm.pth
        │   ├── best_lstm_attention.pth
        │   └── history_*.json               # Loss par époque
        └── plots/                           # Toutes les figures PNG
```

### Scripts `src/` (résumé)

| Script | Rôle |
|--------|------|
| `preprocessing.py` | Lit `*_clean.csv`, ajoute features, écrit `*_with_features.csv`. Option `--all` enchaîne `build_splits_sequences`. |
| `build_splits_sequences.py` | Coupe train/val/test dans le temps, Min-Max sur train, crée fenêtres 168→24, sauve `.pkl`. |
| `models.py` | Définit les 3 architectures PyTorch (`build_model(name, …)`). |
| `train.py` | Boucle d’entraînement Adam + MSE, early stopping, sauve le meilleur poids. |
| `evaluate.py` | Charge checkpoint + `test.pkl`, prédit, affiche métriques, écrit CSV et PNG. |
| `plot_training_history.py` | Lit `history_*.json` → courbes de loss. |
| `diebold_mariano.py` | Compare les erreurs des 3 modèles (test complet). |
| `plot_attention_heatmap.py` | Visualise les poids α sur 168 pas (modèle attention). |
| `generate_report.py` | Regénère `report.html` à partir de métriques et plots. |

### Contenu d’un fichier `.pkl` (train / val / test)

Chaque `.pkl` est un dictionnaire Python :

| Clé | Forme | Signification |
|-----|--------|---------------|
| `X` | `(N, 168, 6)` | Entrée : 168 pas × 6 variables normalisées |
| `y` | `(N, 24)` | Cible : 24 prochaines heures de `PJME_MW` (normalisé) |
| `start_idx` / méta | — | Indices dans la série (selon version du script) |

Les **6 features** (`sequence_meta.json`) : `PJME_MW`, `dow`, `hour`, `is_holiday`, `is_weekend`, `month`.

---

## 4. Après `git pull` : ce qui est sur GitHub ou non

### 4.1 Déjà dans le dépôt (rien à recalculer pour consulter)

| Élément | Emplacement |
|--------|-------------|
| Données CSV (raw, clean, with_features) | `energy_forecast/data/` |
| Scaler + meta | `sequence_scaler.joblib`, `sequence_meta.json` |
| 3 modèles entraînés | `results/checkpoints/best_*.pth` |
| Historiques d’entraînement | `history_*.json` |
| Métriques, DM, rapport, plots | `results/` |

### 4.2 À recréer en local (obligatoire pour `train.py` / `evaluate.py`)

| Fichier | Taille approx. | Commande |
|---------|----------------|----------|
| `train.pkl` | ~457 Mo | `python energy_forecast/src/build_splits_sequences.py` |
| `val.pkl` | ~56 Mo | (idem) |
| `test.pkl` | ~56 Mo | (idem) |

Ces fichiers sont dans **`.gitignore`** : ne jamais les `git add` (rejet GitHub > 100 Mo).

### 4.3 Récupérer le code

```bash
cd "chemin/vers/PJM Hourly Energy Consumption"
git pull origin main
```

Clone initial :

```bash
git clone git@github.com:Andassa/PJM-Hourly-Energy-Consumption.git
cd PJM-Hourly-Energy-Consumption
```

**Git Bash (Windows) :** utiliser `/` dans les chemins (`energy_forecast/src/train.py`), pas `\`.

---

## 5. Installation

À la **racine du dépôt** :

```bash
python -m venv .venv
source .venv/Scripts/activate    # Git Bash (Windows)
# ou : .\.venv\Scripts\Activate.ps1   # PowerShell

python -m pip install -U pip
python -m pip install -r requirements.txt
```

Vérification :

```bash
python -c "import torch, pandas, sklearn, yaml; print('OK')"
```

**GPU (optionnel) :** dans `energy_forecast/config.yaml`, mettre `device: cuda` si PyTorch CUDA est installé ; sinon `cpu` (défaut).

---

## 6. Parcours A — Vérifier le projet (sans réentraîner)

**Durée :** ~10–20 min. **But :** confirmer que votre machine reproduit l’environnement du groupe.

### Étape A.1 — Recréer les `.pkl`

```bash
python energy_forecast/src/build_splits_sequences.py
```

Vous devez voir des chemins vers `train.pkl`, `val.pkl`, `test.pkl`.

### Étape A.2 — Évaluer les 3 modèles (affichage des résultats)

Chaque commande affiche dans le **terminal** un bloc JSON (MAE, RMSE, MAPE), met à jour `metrics.csv` et crée des PNG.

```bash
python energy_forecast/src/evaluate.py --config energy_forecast/config.yaml --checkpoint energy_forecast/results/checkpoints/best_lstm.pth

python energy_forecast/src/evaluate.py --config energy_forecast/config.yaml --checkpoint energy_forecast/results/checkpoints/best_bilstm.pth

python energy_forecast/src/evaluate.py --config energy_forecast/config.yaml --checkpoint energy_forecast/results/checkpoints/best_lstm_attention.pth
```

**Référence** (test complet, `n_points` = 344 376 = 14 349 fenêtres × 24 h) :

| Modèle | MAE (MW) | RMSE (MW) | MAPE (%) |
|--------|----------|-----------|----------|
| LSTM | **1417** | **1982** | **4,50** |
| Bi-LSTM | 1424 | 1999 | 4,51 |
| LSTM + attention | 1457 | 2020 | 4,61 |

Test rapide (moins de données) :

```bash
python energy_forecast/src/evaluate.py --config energy_forecast/config.yaml --checkpoint energy_forecast/results/checkpoints/best_lstm.pth --subset 5000
```

### Étape A.3 — Consulter le rapport

Ouvrir dans un navigateur (double-clic ou glisser dans Chrome/Firefox) :

```
energy_forecast/results/report.html
```

### Étape A.4 — Scripts complémentaires (optionnel)

```bash
python energy_forecast/src/plot_training_history.py --config energy_forecast/config.yaml
python energy_forecast/src/diebold_mariano.py --config energy_forecast/config.yaml
python energy_forecast/src/plot_attention_heatmap.py --config energy_forecast/config.yaml
python energy_forecast/src/generate_report.py --config energy_forecast/config.yaml
```

`diebold_mariano.py` charge les **trois** checkpoints : prévoir quelques minutes en CPU.

### Récap parcours A

```
git pull → venv + pip install → build_splits_sequences.py → evaluate (×3) → report.html
```

---

## 7. Parcours B — Pipeline complet depuis zéro

**Durée :** EDA + preprocessing quelques minutes ; entraînement **plusieurs heures par modèle** en CPU.

### B.1 — Données brutes

#### Où mettre / changer le fichier source

Le pipeline lit **un seul CSV** dans ce dossier :

```
energy_forecast/data/raw/
```

| Action | Détail |
|--------|--------|
| **Fichier utilisé par le groupe (projet actuel)** | `energy_forecast/data/raw/PJME_hourly.csv` |
| **Copie à la racine du dépôt** | `Hourly Energy Consumption/PJME_hourly.csv` — **même contenu** que `data/raw/` (copie Kaggle) ; ce n’est pas ce chemin que les scripts lisent directement |
| **Changer de jeu de données** | Remplacer ou ajouter un `.csv` dans `data/raw/`, puis **refaire** EDA → `preprocessing.py --all` → entraînement (parcours B et C) |

Le notebook EDA cherche en priorité, dans `data/raw/` :

1. `PJME_hourly.csv`
2. `PJM_Load_hourly.csv`
3. sinon le premier `*.csv` trouvé

Pour forcer **PJM Load**, ne laisser que `PJM_Load_hourly.csv` dans `data/raw/` (ou retirer `PJME_hourly.csv` temporairement), puis relancer le notebook.

#### Quelle donnée a servi à **notre** entraînement ?

**En une phrase :** nous avons prédit la consommation électrique horaire de la zone **PJM East (PJME)** entre **2002 et 2018**, à partir du fichier **`PJME_hourly.csv`** (colonne `PJME_MW`), après nettoyage dans le notebook — **pas** la charge totale `PJM_Load_hourly.csv`.

Chaîne réelle (pas de lecture directe du dossier `Hourly Energy Consumption\` par `train.py`) :

```text
data/raw/PJME_hourly.csv
  → EDA → PJME_hourly_clean.csv
  → preprocessing → PJME_hourly_with_features.csv
  → build_splits_sequences → train/val/test.pkl
  → train.py → best_*.pth
```

#### PJME vs PJM Load — comparaison

| | **PJME_hourly.csv** (notre étude) | **PJM_Load_hourly.csv** |
|---|-----------------------------------|-------------------------|
| **Signification** | Charge de la sous-zone **PJM East** | Charge **totale** du réseau PJM (agrégat) |
| **Colonne cible** | `PJME_MW` | `PJM_Load_MW` |
| **Période (fichier du dépôt)** | ~2002 → 2018 (~145 000 h) | ~1999 → 2001 (~33 000 h) |
| **Modèles / métriques actuels** | Oui (`best_*.pth`, MAE ~1417 MW, etc.) | Non — autre série, autre période |

Ce ne sont **pas** les mêmes grandeurs : on ne peut pas « tester » un modèle entraîné sur PJME avec des données PJM Load comme si c’était le même problème.

#### Peut-on utiliser `PJM_Load_hourly.csv` pour entraîner aussi ?

| Situation | Possible ? |
|-----------|------------|
| **Réutiliser** les checkpoints actuels (`best_lstm.pth`, etc.) sur PJM Load | **Non** — autre signal, autre échelle |
| **Comparer** nos MAE/RMSE PJME avec une run sur PJM Load | **Non** — ce n’est pas comparable |
| **Lancer un second projet** sur PJM Load (tout refaire) | **Oui** — copier le CSV dans `data/raw/`, EDA, `--all`, `train.py` × 3, `evaluate.py` |

**Attention si vous passez sur PJM Load :** série **beaucoup plus courte** (~3 ans) → moins de fenêtres d’entraînement, résultats et rapport à refaire entièrement, titre de soutenance à adapter (« charge totale PJM » et non « PJM East »).

**Recommandation pour le rendu du groupe :** rester sur **PJME** ; ne mentionner PJM Load que pour expliquer qu’un autre CSV du jeu Kaggle existe mais n’a pas été utilisé pour les chiffres du rapport.

### B.2 — EDA et nettoyage

```bash
jupyter notebook energy_forecast/notebooks/01_eda_pjm_hourly_load.ipynb
```

Exécuter toutes les cellules. Sortie attendue : `energy_forecast/data/processed/PJME_hourly_clean.csv`.

### B.3 — Features + séquences (une commande)

```bash
python energy_forecast/src/preprocessing.py --all
```

Équivalent à : `preprocessing.py` puis `build_splits_sequences.py`.

### B.4 — Suite

Enchaîner le [parcours C](#8-parcours-c--entraîner-et-évaluer-les-3-modèles) puis les scripts de la section 6 A.4.

---

## 8. Parcours C — Entraîner et évaluer les 3 modèles

**Prérequis :** `train.pkl` et `val.pkl` présents (étape A.1 ou B.3).

Pour **chaque** architecture :

1. Éditer `energy_forecast/config.yaml` — une seule ligne active :

```yaml
model:
  name: lstm          # puis bilstm, puis lstm_attention
```

2. Entraîner :

```bash
python energy_forecast/src/train.py --config energy_forecast/config.yaml
```

**Pendant l’entraînement**, le terminal affiche à chaque époque :

```text
Epoch 001  train_mse=...  val_mse=...
```

À la fin :

```text
Checkpoint : .../best_lstm.pth
Historique : .../history_lstm.json
```

Early stopping peut arrêter avant 100 époques (patience = 10 dans `config.yaml`).

3. Évaluer :

```bash
python energy_forecast/src/evaluate.py --config energy_forecast/config.yaml
```

**Debug rapide** (quelques minutes) :

```bash
python energy_forecast/src/train.py --config energy_forecast/config.yaml --subset 5000 --max-epochs 3
```

| Modèle | `model.name` | Checkpoint | Historique |
|--------|--------------|------------|------------|
| LSTM | `lstm` | `best_lstm.pth` | `history_lstm.json` |
| Bi-LSTM | `bilstm` | `best_bilstm.pth` | `history_bilstm.json` |
| LSTM + attention | `lstm_attention` | `best_lstm_attention.pth` | `history_lstm_attention.json` |

---

## 9. Référence des commandes (entrée → sortie)

Toutes les commandes s’exécutent depuis la **racine du dépôt**.

| Commande | Entrée principale | Affichage terminal | Fichiers créés / mis à jour |
|----------|-------------------|--------------------|-----------------------------|
| `build_splits_sequences.py` | `*_with_features.csv` | Chemins des `.pkl`, shapes | `train/val/test.pkl`, scaler, `sequence_meta.json` |
| `preprocessing.py` | `*_clean.csv` | Colonnes du CSV enrichi | `*_with_features.csv` |
| `preprocessing.py --all` | `*_clean.csv` | Idem + séquences | features CSV + `.pkl` |
| `train.py` | `train.pkl`, `val.pkl`, `config.yaml` | Loss par époque, chemin checkpoint | `best_<model>.pth`, `history_<model>.json` |
| `evaluate.py` | `test.pkl`, checkpoint | **JSON** MAE/RMSE/MAPE | `metrics.csv`, `eval_*_scatter/window/residuals.png` |
| `plot_training_history.py` | `history_*.json` | Chemins des PNG | `loss_lstm.png`, `loss_all_models.png`, … |
| `diebold_mariano.py` | 3 checkpoints, `test.pkl` | DM, p-valeur par paire | `dm_results.csv` |
| `plot_attention_heatmap.py` | `best_lstm_attention.pth` | Chemins figures | `attn_heatmap_*.png` |
| `generate_report.py` | metrics, plots, dm | Chemin `report.html` | `results/report.html` |

---

## 10. Fichier `config.yaml`

Emplacement : `energy_forecast/config.yaml` (chemins relatifs au dossier `energy_forecast/`).

| Section | Rôle |
|---------|------|
| `paths` | Emplacements des `.pkl`, scaler, dossier checkpoints |
| `model.name` | Architecture active pour `train.py` / checkpoint par défaut de `evaluate.py` |
| `model.*` | `hidden_size`, `num_layers`, `dropout`, `horizon` (24), `attn_dim` |
| `training` | `batch_size`, `lr`, `max_epochs`, `patience`, `seed` |
| `device` | `cpu` ou `cuda` |

**Important :** pour évaluer un modèle sans changer `model.name`, utiliser `--checkpoint` (voir section 6).

---

## 11. Roadmap d’interprétation des résultats

Suivre cet ordre pour préparer la soutenance ou le rapport.

### Étape 1 — Métriques chiffrées (`metrics.csv` + terminal)

| Métrique | Signification | Comment parler en soutenance |
|----------|---------------|------------------------------|
| **MAE (MW)** | Erreur absolue moyenne | « En moyenne, le modèle se trompe de X MW » |
| **RMSE (MW)** | Pénalise les grosses erreurs | « Les pics d’erreur tirent RMSE vers le haut » |
| **MAPE (%)** | Erreur relative | « ~4,5 % d’écart relatif sur le test » |
| **n_points** | Nombre de valeurs horaires évaluées | Doit être **344 376** pour le test complet |

**Comparer les 3 modèles :** le LSTM a le MAE le plus bas sur notre run ; l’attention est légèrement moins bonne ici (hypothèses : sur-paramétrisation, même budget d’entraînement).

### Étape 2 — Courbes d’entraînement (`loss_*.png`)

| Observation | Interprétation |
|-------------|----------------|
| Train ↓ et val ↓ puis val stable | Apprentissage sain |
| Val remonte, train continue ↓ | Surapprentissage (early stopping limite ce risque) |
| `loss_all_models.png` | Comparer vitesse de convergence des 3 archi |

Fichiers : `history_*.json` contiennent les valeurs numériques (`train_loss`, `val_loss` par époque).

### Étape 3 — Nuage de points (`eval_*_scatter.png`)

- Axe X = charge **observée**, axe Y = **prédite** (MW).
- Points proches de la diagonale rouge **y = x** → bonnes prédictions.
- Nuage élargi aux fortes charges → erreurs plus grandes aux pics (typique charge électrique).

### Étape 4 — Fenêtre 24 h (`eval_*_window_<idx>.png`)

- Une **fenêtre** du test : courbe observée vs prédite sur **24 heures**.
- Utile pour voir si le modèle suit le **profil journalier** et les pics.
- L’index `<idx>` change à chaque run (tirage pseudo-aléatoire seed 42).

### Étape 5 — Résidus (`eval_*_residuals.png`)

- Histogramme de **prédit − observé** (MW).
- Centré autour de 0 → pas de biais systématique fort.
- Queue épaisse → quelques grosses erreurs.

### Étape 6 — Diebold–Mariano (`dm_results.csv`)

| Colonne | Sens |
|---------|------|
| `model_a`, `model_b` | Paire comparée |
| `dm_stat`, `p_value` | Test si la différence de loss MSE est significative |
| `significant_5pct` | `True` si p < 0,05 |
| `interpretation` | Phrase lisible |

**Résultats actuels du projet :**

- LSTM **significativement** meilleur que Bi-LSTM et que LSTM+attention (MSE sur le test).
- Bi-LSTM significativement meilleur que LSTM+attention.

→ Les écarts de MAE ne sont pas seulement dus au hasard d’échantillonnage (sous hypothèses du test DM).

### Étape 7 — Attention (`attn_heatmap_*.png`)

- **Une fenêtre** : quels pas parmi les 168 h le modèle « regarde » pour prédire 24 h.
- **Moyenne sur 200 fenêtres** : comportement typique.
- Pics sur heures récentes ou sur motifs répétés → interprétation physique à discuter en groupe.

### Étape 8 — Rapport HTML (`report.html`)

Synthèse des sections 1–7 pour l’encadrant. Régénérer après tout changement de métriques ou figures :

```bash
python energy_forecast/src/generate_report.py --config energy_forecast/config.yaml
```

---

## 12. Catalogue des fichiers de résultats

### `results/metrics.csv`

| Colonne | Description |
|---------|-------------|
| `model` | `lstm`, `bilstm`, `lstm_attention` |
| `checkpoint` | Chemin du `.pth` utilisé |
| `mae_mw`, `rmse_mw`, `mape_pct` | Métriques sur le test (MW / %) |
| `n_points` | 344 376 = évaluation complète |

### `results/dm_results.csv`

Tableau des tests DM entre paires de modèles (voir section 11.6).

### `results/checkpoints/best_*.pth`

Checkpoint PyTorch : poids du modèle au meilleur `val_mse`, plus métadonnées (`model_name`, hyperparamètres).

### `results/checkpoints/history_*.json`

Exemple de structure :

```json
{
  "train_loss": [ ... ],
  "val_loss": [ ... ],
  "best_epoch": 42
}
```

### `results/plots/` — liste des figures

| Fichier (pattern) | Produit par | À quoi ça sert |
|-------------------|-------------|----------------|
| `loss_lstm.png`, `loss_bilstm.png`, `loss_lstm_attention.png` | `plot_training_history.py` | Loss train/val par époque |
| `loss_all_models.png` | idem | Comparaison des 3 courbes |
| `eval_best_lstm_scatter.png` (etc.) | `evaluate.py` | Observé vs prédit |
| `eval_best_*_window_*.png` | `evaluate.py` | Exemple 24 h |
| `eval_best_*_residuals.png` | `evaluate.py` | Distribution des erreurs |
| `attn_heatmap_window_*.png` | `plot_attention_heatmap.py` | Attention sur un exemple |
| `attn_heatmap_mean_*windows.png` | idem | Attention moyennée |

### `results/report.html`

Rapport web autonome : contexte, hyperparamètres, tableaux, figures intégrées. **Ouvrir localement** (pas besoin de serveur).

### `data/processed/sequence_meta.json`

Référence pour la soutenance : dates exactes train/val/test, nombre de fenêtres, liste des features.

Exemple (périodes test) :

- **Test :** 2016-12-05 → 2018-08-03  
- **Fenêtres test :** 14 349 séquences de 168 h

---

## 13. Dépannage

| Problème | Solution |
|----------|----------|
| `ModuleNotFoundError: torch` | Activer `.venv`, `pip install -r requirements.txt` |
| `Aucun fichier *_clean.csv` | Lancer le notebook EDA puis `preprocessing.py --all` |
| `Aucun *_with_features.csv` | `python energy_forecast/src/preprocessing.py` |
| `FileNotFoundError` sur `train.pkl` | `python energy_forecast/src/build_splits_sequences.py` |
| Chemin invalide (Git Bash) | `energy_forecast/src/train.py` avec `/` |
| Mauvais modèle évalué | Utiliser `--checkpoint` explicite |
| Push rejeté (> 100 Mo) | Ne pas committer les `.pkl` (déjà dans `.gitignore`) |
| `Co-authored-by: Cursor` sur GitHub | Désactiver **Commit attribution** dans Cursor ; réécrire le message de commit si besoin |
| Métriques très différentes des références | Vérifier `n_points` (subset ?), même `.pkl`/scaler, même checkpoint |
| Confusion PJME / PJM Load | Voir [B.1 — Données brutes](#b1--données-brutes) ; le projet livré utilise **PJME** uniquement |

---

## 14. Liens utiles

| Document | Contenu |
|----------|---------|
| [README.md](README.md) | Objectifs, maths, architecture RNA, répartition modules |
| [README.md — Exécution](README.md#execution-avancement) | Tableau d’avancement du projet |
| `energy_forecast/results/report.html` | Livrable visuel |
| `energy_forecast/config.yaml` | Hyperparamètres actifs |

---

## Récapitulatif ultra-court

| Besoin | Commande / fichier |
|--------|-------------------|
| Installer | `pip install -r requirements.txt` |
| Préparer les tenseurs | `python energy_forecast/src/build_splits_sequences.py` |
| Voir les chiffres | `evaluate.py` → terminal + `metrics.csv` |
| Voir les graphiques | `results/plots/` + `report.html` |
| Réentraîner | `train.py` × 3 (`config.yaml`) |
| Comparer statistiquement | `diebold_mariano.py` → `dm_results.csv` |
| Livrable encadrant | `results/report.html` (+ PDF export navigateur) |

**En une phrase :** ce dépôt prédit 24 h de charge PJME à partir de 168 h d’historique ; après `pull` + `.pkl`, ouvrez `report.html` et utilisez la [section 11](#11-roadmap-dinterprétation-des-résultats) pour interpréter chaque résultat.

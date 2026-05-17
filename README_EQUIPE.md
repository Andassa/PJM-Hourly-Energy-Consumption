# Guide équipe — mise à jour, vérification et reproduction

Ce document complète le [README principal](README.md). Il explique **quoi faire après un `git pull`** pour retrouver le même environnement que sur la machine qui a entraîné les modèles.

**Dépôt :** [PJM-Hourly-Energy-Consumption](https://github.com/Andassa/PJM-Hourly-Energy-Consumption)

---

## Ce que vous récupérez avec `git pull`

Après `git pull`, le dépôt contient déjà :

| Élément | Emplacement |
|--------|-------------|
| Données brutes, nettoyées, avec features | `energy_forecast/data/raw/`, `energy_forecast/data/processed/*.csv` |
| Scaler Min-Max + métadonnées des fenêtres | `sequence_scaler.joblib`, `sequence_meta.json` |
| **3 modèles entraînés** (LSTM, Bi-LSTM, attention) | `energy_forecast/results/checkpoints/best_*.pth` |
| Historiques d’entraînement | `history_*.json` |
| Métriques test finales | `energy_forecast/results/metrics.csv` |
| Graphiques + rapport HTML | `energy_forecast/results/plots/`, `report.html` |
| Test Diebold–Mariano | `energy_forecast/results/dm_results.csv` |
| Code (train, evaluate, rapport, DM, heatmaps) | `energy_forecast/src/` |

### Ce qui n’est **pas** sur GitHub

Les fichiers suivants sont **trop volumineux** pour GitHub (limite 100 Mo) :

- `energy_forecast/data/processed/train.pkl`
- `energy_forecast/data/processed/val.pkl`
- `energy_forecast/data/processed/test.pkl`

Ils doivent être **recréés en local** (environ 5–15 minutes). Sans eux, `train.py` et `evaluate.py` ne peuvent pas tourner.

---

## Prérequis

- **Python 3.10+**
- **Git**
- Sous Windows : de préférence **Git Bash** pour les commandes ci-dessous
- Utiliser des **slashes** `/` dans les chemins : `energy_forecast/src/train.py` (pas `\`)

---

## 1. Récupérer le projet

```bash
cd "chemin/vers/PJM Hourly Energy Consumption"
git pull origin main
```

Si vous n’avez pas encore cloné :

```bash
git clone git@github.com:Andassa/PJM-Hourly-Energy-Consumption.git
cd PJM-Hourly-Energy-Consumption
```

---

## 2. Installation (une fois par machine)

À la **racine du dépôt** (là où se trouve `requirements.txt`) :

```bash
python -m venv .venv
source .venv/Scripts/activate    # Git Bash (Windows)
# ou : .\.venv\Scripts\Activate.ps1   # PowerShell

python -m pip install -U pip
python -m pip install -r requirements.txt
```

Vérification rapide :

```bash
python -c "import torch, pandas, sklearn, yaml; print('OK')"
```

---

## 3. Parcours A — Vérifier le projet (recommandé)

**Durée indicative :** 10–20 minutes (sans réentraînement).

**Objectif :** recréer les `.pkl`, vérifier que l’évaluation retrouve des métriques proches de `metrics.csv`, consulter le rapport.

### Étape A.1 — Recréer les séquences (obligatoire)

Les CSV avec features sont déjà dans le dépôt. Il suffit de reconstruire les fenêtres 168 h → 24 h :

```bash
python energy_forecast/src/build_splits_sequences.py
```

Fichiers créés :

- `train.pkl`, `val.pkl`, `test.pkl`
- `sequence_scaler.joblib` et `sequence_meta.json` (écrasés / mis à jour)

> Variante si vous devez tout refaire depuis le CSV nettoyé :  
> `python energy_forecast/src/preprocessing.py --all`  
> (features + splits + `.pkl` en une commande)

### Étape A.2 — Vérifier l’évaluation (optionnel mais conseillé)

Par défaut, `evaluate.py` utilise le modèle indiqué dans `config.yaml` (`model.name`). Pour vérifier le **LSTM** (meilleur sur le test) :

1. Ouvrir `energy_forecast/config.yaml`
2. Mettre par exemple : `name: lstm` (décommenter la bonne ligne)
3. Lancer :

```bash
python energy_forecast/src/evaluate.py --config energy_forecast/config.yaml
```

Comparer avec `energy_forecast/results/metrics.csv` (test complet, `n_points` ≈ 344 376) :

| Modèle | MAE (MW) | RMSE (MW) | MAPE (%) |
|--------|----------|-----------|----------|
| LSTM | **1417** | **1982** | **4,50** |
| Bi-LSTM | 1424 | 1999 | 4,51 |
| LSTM + attention | 1457 | 2020 | 4,61 |

De petites différences peuvent apparaître si le scaler ou les `.pkl` ont été régénérés avec une autre version de librairie ; l’ordre de grandeur doit rester le même.

### Étape A.3 — Consulter le livrable

Ouvrir dans un navigateur :

```
energy_forecast/results/report.html
```

Les images du rapport sont dans `energy_forecast/results/plots/` (chemins relatifs).

### Étape A.4 — Scripts optionnels

```bash
python energy_forecast/src/plot_training_history.py --config energy_forecast/config.yaml
python energy_forecast/src/diebold_mariano.py --config energy_forecast/config.yaml
python energy_forecast/src/plot_attention_heatmap.py --config energy_forecast/config.yaml
python energy_forecast/src/generate_report.py --config energy_forecast/config.yaml
```

`diebold_mariano.py` charge les **trois** checkpoints et peut prendre quelques minutes sur CPU.

---

## 4. Parcours B — Réentraîner un modèle

**Durée indicative :** plusieurs heures par architecture en **CPU** (selon la machine).

À n’utiliser que si vous voulez **refaire l’entraînement** vous-même, pas pour simplement consulter les résultats du groupe.

1. Terminer le **parcours A**, étape A.1 (`build_splits_sequences.py`).
2. Dans `energy_forecast/config.yaml`, choisir **une** architecture :
   - `name: lstm`
   - `name: bilstm`
   - `name: lstm_attention`
3. Lancer :

```bash
python energy_forecast/src/train.py --config energy_forecast/config.yaml
python energy_forecast/src/evaluate.py --config energy_forecast/config.yaml
```

4. Répéter en changeant `model.name` pour les deux autres modèles si besoin.

Les checkpoints sont écrits dans :

`energy_forecast/results/checkpoints/best_<nom>.pth`

Test rapide (debug) :

```bash
python energy_forecast/src/train.py --config energy_forecast/config.yaml --subset 5000 --max-epochs 3
```

---

## 5. Notebook EDA

Le notebook `energy_forecast/notebooks/01_eda_pjm_hourly_load.ipynb` produit notamment `PJME_hourly_clean.csv`.

**Après un `git pull` normal**, ce CSV est **déjà présent** : vous n’avez pas besoin de relancer l’EDA pour le parcours A.

Relancer l’EDA seulement si :

- le fichier `*_clean.csv` manque, ou
- vous modifiez le nettoyage (DST, trous, etc.).

Ensuite :

```bash
python energy_forecast/src/preprocessing.py --all
```

---

## 6. Dépannage

| Problème | Solution |
|----------|----------|
| `ModuleNotFoundError: torch` (ou autre) | Activer `.venv` puis `pip install -r requirements.txt` |
| `Aucun fichier *_clean.csv` | Notebook EDA puis `preprocessing.py --all` |
| `Aucun *_with_features.csv` | `python energy_forecast/src/preprocessing.py` |
| `FileNotFoundError` sur `train.pkl` | `python energy_forecast/src/build_splits_sequences.py` |
| Chemin invalide sous Git Bash | Utiliser `/` : `energy_forecast/src/train.py` |
| `evaluate.py` évalue le mauvais modèle | Vérifier `model.name` dans `config.yaml` |
| Push rejeté (fichier > 100 Mo) | Ne **jamais** committer les `.pkl` ; ils sont dans `.gitignore` |

---

## 7. Récapitulatif en une ligne

**Pull → venv + `pip install` → `build_splits_sequences.py` → ouvrir `report.html`**  
= vous avez le même projet que l’équipe, sans réentraîner.

**Réentraînement** = parcours B, uniquement si nécessaire.

---

## 8. Liens utiles

- [README principal](README.md) — contexte, maths, architecture des modèles
- [Exécution et avancement](README.md#execution-avancement) — chaîne de commandes complète
- Rapport : `energy_forecast/results/report.html`

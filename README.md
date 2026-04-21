#  AI-Driven Malware — TP1 Intelligence Artificielle & Cybersécurité

> **SUP DE VINCI** — TP1 IA & Cybersécurité  



##  Structure du projet

```
ai_keylogger/
├── keylogger.py           ← Partie I  : Capture des frappes clavier
├── sentiment_analyzer.py  ← Partie II : Analyse de sentiments (VADER)
├── anomaly_detector.py    ← Partie II : Détection d'anomalies (Isolation Forest)
├── sensitive_detector.py  ← Partie III: Classification données sensibles (Regex + RF)
├── report_generator.py    ← Partie IV : Rapports HTML + visualisations Plotly
├── extension/
│   ├── __init__.py
│   ├── app_context.py     ← Extension B : Contexte applicatif
│   └── encryption.py      ← Extension C : Chiffrement AES-256-GCM
├── data/                  ← Logs, modèles, exports (gitignored)
│   ├── log.txt
│   ├── sentiments.json
│   ├── alerts.json
│   ├── detections.json
│   └── report.html
├── tests/
│   └── test_all.py        ← Tests unitaires (pytest)
├── requirements.txt
└── README.md
```

---

##  Installation

```bash
# 1. Cloner le repo
git clone https://github.com/Ecknard/AI_K.git
cd ai_keylogger

# 2. Création de l'environnement virtuel
python -m venv env

# 3. Activer l'environnement
# Windows :
.\env\Scripts\activate
# Linux / macOS :
source env/bin/activate

# 4. Installation les dépendances
pip install -r requirements.txt

# 5. Vérifier l'installation
python -c "import pynput; print('pynput OK')"
python -c "import sklearn; print('sklearn OK')"
python -c "import plotly; print('plotly OK')"
```

---

##  Utilisation

### Partie I — Lancer le keylogger de base
```bash
python keylogger.py
# → Capture les frappes et sauvegarde dans data/log.txt toutes les 10s
# → Ctrl+C pour arrêter
```

### Partie II — Tester l'analyse de sentiments
```bash
python sentiment_analyzer.py
```

### Partie II — Tester la détection d'anomalies
```bash
python anomaly_detector.py
```

### Partie III — Tester la détection de données sensibles
```bash
python sensitive_detector.py
```

### Partie IV — Générer le rapport HTML
```bash
python report_generator.py
# → data/report.html (ouvrir dans un navigateur)
```

### Tests unitaires
```bash
python -m pytest tests/ -v
# ou
python tests/test_all.py
```

---

##  Réponses aux questions du TP

### Tâche 1 — Analyse conceptuelle

#### Question 1.1 — Définition et dangers

Un **keylogger** est un programme qui intercepte et enregistre les événements clavier d'un système à l'insu de l'utilisateur.

| Axe | Danger concret |
|-----|---------------|
| **Données visées** | Mots de passe, coordonnées bancaires, messages privés, identifiants professionnels |
| **Vecteurs d'infection** | Email de phishing, logiciel piraté, clé USB infectée, exploit navigateur |
| **Persistance** | Clé de registre (`HKCU\Run`), cron job, service système, rootkit |
| **Exfiltration** | Email SMTP automatique, upload HTTP/FTP, webhook Discord/Telegram |

#### Question 1.2 — Usages légitimes

| Secteur | Usage | Condition légale |
|---------|-------|-----------------|
| Entreprise | Audit sécurité interne, investigation d'incident | Consentement écrit, charte informatique, information des employés (RGPD art. 13) |
| Parentalité | Contrôle parental enfant mineur | Autorité parentale légale, enfant sous 15 ans |
| Forensic/SOC | Investigation post-incident | Mandat judiciaire ou autorisation hiérarchique documentée |

---

### Tâche 3 — Questions d'analyse de code

#### Question 3.2 — Le paramètre `on_press`
`on_press` est un **callback** (fonction de rappel) passé au Listener. pynput l'appelle automatiquement dans son thread interne à chaque événement clavier. La valeur passée est une **référence à la fonction** (sans parenthèses) : `on_press=processkeys`.

#### Question 3.3 — `with` et `join()`
- `with keyboard_listener:` — le gestionnaire de contexte appelle automatiquement `listener.start()` à l'entrée et `listener.stop()` à la sortie (même en cas d'exception).
- `keyboard_listener.join()` — **bloque le thread principal** jusqu'à ce que le listener s'arrête. Sans `join()`, le programme se terminerait immédiatement après la ligne `with`, arrêtant le listener avant toute capture.

---

### Tâche 4 — Question 4.2 — Bloc try/except

Les touches spéciales (Ctrl, Alt, F1…) sont des instances de `pynput.keyboard.Key` et **n'ont pas d'attribut `.char`**. Tenter d'accéder à `key.char` lève une `AttributeError`. Le bloc `try/except AttributeError` permet de distinguer les deux cas :

```python
try:
    char = key.char       # Touche alphanumérique → .char existe
except AttributeError:
    # Touche spéciale → .char n'existe pas, traitement séparé
```

---

### Tâche 5 — Questions

#### Question 5.2 — Modes de fichier
| Mode | Comportement |
|------|-------------|
| `'a'` | Append : **ajoute à la fin** sans écraser. Crée le fichier si inexistant. |
| `'w'` | Write : **écrase tout le contenu** existant. |

→ On utilise `'a'` pour ne pas perdre les logs précédents entre deux appels à `report()`.

**Pourquoi vider `log` après écriture ?** Pour éviter d'écrire les mêmes frappes à chaque intervalle suivant (duplication des données).

**Pourquoi fermer le fichier ?** Pour vider le buffer OS (flush), libérer le descripteur de fichier, et garantir qu'un autre processus peut lire le fichier.

**Pourquoi `global log, path` ?** Python cherche les variables en local par défaut. Sans `global`, une assignation comme `log = ""` créerait une variable locale sans modifier la globale.

#### Question 5.3 — Timer auto-relancé

```python
def report(interval=10):
    # ... écriture ...
    timer = threading.Timer(interval, report, args=[interval])
    timer.daemon = True
    timer.start()  # Se relance lui-même → boucle infinie non-bloquante
```

#### Question 5.4 — Points faibles

| Dimension | Point faible | Amélioration |
|-----------|-------------|-------------|
| Détection | pynput a une signature connue des AV | Obfuscation, hooking bas niveau |
| Lisibilité | log.txt brut difficile à analyser | JSON horodaté structuré (Partie II) |
| Contexte | Aucune info sur l'app active | Extension B (pygetwindow) |
| Horodatage | Frappes non horodatées individuellement | Méta-données timestamp (Tâche 7.1) |
| Sécurité log | Fichier en clair | Chiffrement AES-GCM (Extension C) |

---

### Tâche 6 — Choix VADER

**Raison** : VADER (Valence Aware Dictionary and sEntiment Reasoner) est spécifiquement conçu pour les textes courts et informels (messages, saisies clavier). Il ne nécessite aucun entraînement, fonctionne en temps réel, et retourne un score `compound` normalisé entre -1 et +1 directement exploitable.

**Limitation** : anglais uniquement → pour le français, voir Extension E (CamemBERT).

---

### Tâche 7 — Questions ML

**Collecte** : 30 à 60 minutes de frappe normale pour établir un profil fiable.

**Normalisation** : Les features ont des échelles très différentes (délais en secondes vs ratios 0–1). Sans normalisation, les features à grande échelle domineraient la distance et biaisent le modèle.

**Contamination Isolation Forest** : Commencer à 0.05 (5%), ajuster selon le taux de faux positifs observé. Un taux trop élevé génère trop d'alertes inutiles.

---

### Tâche 8 — Questions ML classification

**Choix Random Forest** : gère naturellement les features mixtes (continues + binaires), robuste aux outliers, permet d'inspecter `feature_importances_`. Naive Bayes suppose l'indépendance des features (incorrecte ici). SVM est efficace mais nécessite un scaling rigoureux.

**Métrique prioritaire** : **Rappel (recall)**. Un faux négatif (donnée sensible non détectée) est bien plus coûteux qu'un faux positif (mot ordinaire signalé comme sensible).

---

##  Architecture de sécurité

```
Frappes → processkeys() → log buffer
                       ↓
              [every 10s] report()
                       ↓
          sensitive_detector → masquage
                       ↓
          [optionnel] encrypt_file (AES-256-GCM)
                       ↓
                   data/log.enc
```

---

##  Contribution

1. Samuel MENSAH
2. Josiane EGLA

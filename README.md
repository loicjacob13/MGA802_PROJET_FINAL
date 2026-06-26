# MGA802 — Simulateur de Premier League (Groupe 2)

Simulateur de saison de Premier League par modèle de Poisson et Monte-Carlo.  
Auteurs : Fabien · Loïc · Guillaume

---

## Description

Le programme apprend les forces offensives et défensives de chaque équipe à
partir des données historiques (CSV), puis simule N saisons complètes par
tirage de loi de Poisson. Pour chaque équipe il produit :

- probabilité d'être champion
- probabilité de finir dans le top 4 (Ligue des Champions)
- probabilité de relégation
- position moyenne et points moyens sur les N simulations

Un facteur de « moral » est activé à partir de la journée 35 : les équipes
en zone de relégation reçoivent +10 % de motivation, celles du top 4 +5 %.

---

## Installation

```bash
pip install -r requirements.txt
```

Dépendances : `numpy`, `pandas`, `matplotlib`, `scipy`, `sphinx`, `pytest`.

---

## Lancement

> Exécuter depuis la racine du projet (`MGA802_PROJET_FINAL/`).

```bash
python -m simulation.main_NEW
```

Le programme demande :
1. La saison à simuler (ex. `2023-2024`)
2. Le nombre de simulations Monte-Carlo (ex. `500`)
3. Le sigle de l'équipe à détailler (ex. `ARS`)

Le temps d'exécution est affiché automatiquement à la fin de la simulation
(temps total et temps par simulation).

---

## Structure du projet

```
MGA802_PROJET_FINAL/
├── donnees/                     # Chargement et nettoyage des CSV
│   ├── __init__.py
│   ├── donnees.py
│   └── exploration_donnees.py
├── simulation/                  # Modèle de Poisson et simulateur Monte-Carlo
│   ├── __init__.py
│   ├── main_NEW.py              ← point d'entrée principal
│   ├── modele.py
│   ├── simulateur.py
│   ├── forces_promus.py
│   └── recherche_ponderation.py
├── visualisation/               # Graphiques Matplotlib
│   ├── __init__.py
│   └── visualisation.py
├── docs/                        # Documentation Sphinx
│   └── build/html/index.html   ← documentation générée
├── controle_input.py            # Saisies utilisateur sécurisées
├── mesure_temps.py              # Chronomètre de simulation
├── test_donnees.py              # Tests unitaires (pytest)
├── requirements.txt
└── CAS_TEST.md                  ← exemple d'exécution reproductible
```

---

## Performances

Mesures réalisées sur la saison **2023-2024** avec `mesure_performances.py`.

| N simulations | Temps total | Temps / simulation |
|:---:|---:|---:|
| 100 | 6,89 s | 68,9 ms |
| 500 | 34,37 s | 68,7 ms |
| 1 000 | 69,13 s | 69,1 ms |

**Impact du facteur de moral** (actif à partir de la journée 35) :

| N simulations | Moral activé | Moral désactivé | Surcoût |
|:---:|---:|---:|---:|
| 100 | 6,89 s | 3,65 s | +89 % |
| 500 | 34,37 s | 18,09 s | +90 % |
| 1 000 | 69,13 s | 35,31 s | +96 % |

Le facteur de moral double environ le temps de simulation, car il trie le
classement à chaque fin de journée (38 fois par saison simulée).

---

## Cas test reproductible

Voir [CAS_TEST.md](CAS_TEST.md) pour un exemple complet avec les entrées
exactes et les sorties attendues.

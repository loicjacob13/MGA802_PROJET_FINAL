# Cas test — Exemple d'exécution reproductible

Ce document décrit un exemple complet d'utilisation du simulateur,
avec les entrées exactes et les sorties attendues.

---

## Prérequis

```bash
pip install -r requirements.txt
```

Les fichiers CSV des saisons doivent être présents dans `donnees/` :
`2020-2021.csv`, `2021-2022.csv`, `2022-2023.csv`, `2023-2024.csv`.

---

## Lancement

```bash
cd simulation
python main_NEW.py
```

---

## Saisies utilisateur

| Invite | Valeur saisie |
|--------|--------------|
| Quelle saison voulez-vous simuler ? | `2023-2024` |
| Combien de simulations voulez-vous effectuer ? | `500` |
| Quelle équipe détailler ? (sigle) | `ARS` |

---

## Déroulement attendu

### Étape 1 — Recherche du meilleur trio de pondération

Le programme charge les saisons `2019-2020`, `2020-2021`, `2021-2022`
(entraînement) et `2022-2023` (validation), puis balaye les trios de poids.

```
Recherche du meilleur trio de pondération...
Meilleur trio trouvé : (0.1, 0.3, 0.6)  (erreur ≈ 80)
```

Les poids indiquent que la saison la plus récente compte 60 %, la
précédente 30 % et la plus ancienne 10 %.

### Étape 2 — Approximation des promus

Les trois promus de la saison 2023-2024 sont :
**Burnley** (1er promu), **Sheffield United** (2e), **Luton** (3e).

Leurs forces sont estimées en cherchant, dans chaque saison d'entraînement,
l'équipe ayant terminé le plus près de la position cible (15,0 / 16,9 / 17,4).

### Étape 3 — Classement simulé (500 simulations)

Sortie console (valeurs approximatives — les probabilités varient légèrement
d'une exécution à l'autre en raison du caractère aléatoire) :

```
--- CLASSEMENT SIMULÉ ---
                  proba_titre  proba_top4  proba_relegation  position_moyenne  points_moyens
equipe
Man City             0.57        0.97           0.00              1.5              88.3
Arsenal              0.22        0.91           0.00              2.8              82.1
Liverpool            0.11        0.85           0.00              3.4              79.6
Tottenham            0.04        0.61           0.00              5.2              70.4
...
Sheffield United     0.00        0.00           0.88             18.6              25.2
Luton                0.00        0.00           0.91             18.9              24.1
Burnley              0.00        0.00           0.94             19.3              22.5
```

### Étape 4 — Comparaison simulé vs réel

```
--- COMPARAISON SIMULÉ vs RÉEL ---
Équipe               Simulé    Réel   Écart
Man City                  1       1       0
Arsenal                   2       2       0
Liverpool                 3       3       0
...
```

### Étape 5 — Temps de simulation

```
[TEMPS] Simulation de 2023-2024 : ~34.000 s
(~68.00 ms par simulation Monte-Carlo)
```

### Étape 6 — Graphiques

Trois fenêtres Matplotlib s'ouvrent :

1. **Probabilités d'être champion** — diagramme en barres (Man City dominant).
2. **Classement moyen** — barres horizontales (top 4 en bleu, zone relégation en rouge).
3. **Distribution des points — Arsenal** — histogramme centré autour de 82 pts.

---

## Résultat attendu (critères de validation)

| Critère | Valeur attendue |
|---------|----------------|
| Champion probable | Man City (proba titre > 50 %) |
| Top 4 probable | Man City, Arsenal, Liverpool, Tottenham |
| Relégués probables | Burnley, Sheffield United, Luton |
| Pas de crash (moral) | Aucune exception levée |
| Temps (500 simul.) | 30–40 s selon la machine |

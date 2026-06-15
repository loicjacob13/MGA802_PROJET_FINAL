import numpy as np
from donnees import ChargeurDonnees
from modele import ModelePoisson
from simulateur import Simulateur
from visualisation import Visualiseur
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. CHARGEMENT SÉPARÉ DES 3 SAISONS
# ------------------------------------------------------------------
c1 = ChargeurDonnees("2023-2024.csv")
c2 = ChargeurDonnees("2024-2025.csv")
c3 = ChargeurDonnees("2025-2026.csv")
c1.nettoyer()
c2.nettoyer()
c3.nettoyer()

# ------------------------------------------------------------------
# 2. PONDÉRATION TEMPORELLE (décroissance géométrique 1 : 2 : 4)
#    2023-2024 → poids 1/7 ≈ 14%
#    2024-2025 → poids 2/7 ≈ 29%
#    2025-2026 → poids 4/7 ≈ 57%
# ------------------------------------------------------------------
matchs_1 = c1.get_matchs()
matchs_2 = c2.get_matchs()
matchs_3 = c3.get_matchs()

# 1140 matchs uniques — pas de répétition
matchs_tous = np.vstack([matchs_1, matchs_2, matchs_3])

# Vecteur de poids : une valeur par match
poids_1 = np.ones(len(matchs_1)) * (1/7)
poids_2 = np.ones(len(matchs_2)) * (2/7)
poids_3 = np.ones(len(matchs_3)) * (4/7)
poids   = np.concatenate([poids_1, poids_2, poids_3])

print(f"Matchs totaux : {len(matchs_tous)}")   # doit afficher 1140

# ------------------------------------------------------------------
# 3. ENTRAÎNEMENT DU MODÈLE
# ------------------------------------------------------------------
index_equipes = c3.get_index_equipes()
n_equipes     = len(index_equipes)

modele = ModelePoisson(n_equipes)
modele.entrainer(matchs_tous, poids=poids)

# Vérification : affiche les forces et le lambda moyen par équipe
forces   = modele.get_forces()
avantage = modele.get_avantage_domicile()

print(f"\nAvantage domicile appris : {avantage:.3f}")
print(f"\n{'Équipe':<20} {'Attaque':>10} {'Défense':>10} {'Lambda dom':>12}")
for nom, idx in sorted(index_equipes.items()):
    att, def_ = forces[idx]
    lam = np.exp(att - def_ + avantage)
    print(f"{nom:<20} {att:>10.3f} {def_:>10.3f} {lam:>12.2f}")

# ------------------------------------------------------------------
# 4. SIMULATION MONTE-CARLO
# ------------------------------------------------------------------
sim      = Simulateur(forces, avantage, index_equipes)
resultats = sim.simuler_monte_carlo(n_simulations=500)

print("\n--- Résultats Monte-Carlo ---")
print(resultats.to_string())

# ------------------------------------------------------------------
# 5. GRAPHIQUES (Les fenêtres s'ouvrent simultanément)
# ------------------------------------------------------------------
visu = Visualiseur(resultats)
visu.graphique_probabilites_titre()
visu.graphique_classement_moyen()
visu.graphique_distribution_points("Arsenal", simulateur=sim)

plt.show()
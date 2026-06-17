import numpy as np
from donnees import ChargeurDonnees
from modele import ModelePoisson
from simulateur import Simulateur
from visualisation import Visualiseur
import matplotlib.pyplot as plt
from forces_promus import forces_pour_position_cible



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
poids_1 = np.ones(len(matchs_1)) * (1/6)
poids_2 = np.ones(len(matchs_2)) * (2/6)
poids_3 = np.ones(len(matchs_3)) * (3/6)
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

print(f"\n{'Équipe':<20} {'Attaque':>10} {'Défense':>10} {'Avantage DOM':>14} {'Lambda dom':>12}")
print("-" * 70)
for nom, idx in sorted(index_equipes.items()):
    att, def_ = forces[idx]
    av  = avantage[idx]
    lam = np.exp(att - def_ + av)  # lambda contre une équipe de force moyenne (def=0)
    print(f"{nom:<20} {att:>10.3f} {def_:>10.3f} {av:>14.3f} {lam:>12.2f}")

# ------------------------------------------------------------------
# 4. SIMULATION MONTE-CARLO
# ------------------------------------------------------------------
sim      = Simulateur(forces, avantage, index_equipes)
resultats = sim.simuler_monte_carlo(n_simulations=100)


#test de 1 saison juste.
#resultats = sim.simuler_monte_carlo(n_simulations=1)

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


##nouvelles partie
# ----------------------------------------------------------------------
# 1. TROUVER LES 3 ÉQUIPES RELÉGUÉES (les 3 dernières de 2025/2026)
# ----------------------------------------------------------------------
# 'resultats' vient de ta simulation Monte-Carlo, il est trié par position_moyenne.
# Mais pour les VRAIS relégués, on regarde plutôt le vrai classement de la dernière
# saison. Ici on suppose que tu connais les 3 relégués de 2025/2026 :
reLegues = ["West Ham", "Burnley", "Wolves"]  # les 3 reléguées de 2025/2026

# ----------------------------------------------------------------------
# 2. LES 3 PROMUS ET LEUR POSITION CIBLE
# ----------------------------------------------------------------------
promus = [
    ("Coventry", 15.0),  # 1er entrant (champion de Championship)
    ("Ipswich", 16.9),  # 2e entrant
    ("Millwall", 17.4),  # 3e entrant
]

# ----------------------------------------------------------------------
# 3. CONSTRUCTION DES FORCES ET DE L'INDEX DE LA NOUVELLE SAISON
# ----------------------------------------------------------------------
# On part des forces existantes (dict {index: (attaque, defense)}) et de l'index
# {nom: numéro}. On reconstruit deux dicts propres pour les 20 équipes finales.

# ----------------------------------------------------------------------
# 3. CONSTRUCTION DES FORCES, DE L'AVANTAGE ET DE L'INDEX (CORRIGÉ)
# ----------------------------------------------------------------------
# Dictionnaire inverse {index: nom} pour retrouver les noms
nom_de_index = {idx: nom for nom, idx in index_equipes.items()}

# On garde les équipes existantes SAUF les reléguées
forces_saison = {}      # {nouveau_numéro: (attaque, defense)}
avantage_saison = {}    # {nouveau_numéro: avantage domicile}   <-- AJOUTÉ
index_saison = {}       # {nom: nouveau_numéro}
numero = 0              # compteur de numéro pour les nouvelles équipes

for ancien_idx, (att, defe) in forces.items():
    nom = nom_de_index[ancien_idx]      # nom de cette équipe
    if nom in reLegues:                 # on saute les reléguées
        continue
    forces_saison[numero] = (att, defe)         # on garde ses forces
    avantage_saison[numero] = avantage[ancien_idx]  # on garde SON avantage  <-- AJOUTÉ
    index_saison[nom] = numero          # on l'ajoute à l'index
    numero += 1

# On ajoute maintenant les 3 promus avec leurs forces approximées
for nom, position_cible in promus:
    att, defe = forces_pour_position_cible(position_cible, resultats, forces, index_equipes)
    forces_saison[numero] = (att, defe)
    # Pour l'avantage du promu : on prend l'avantage MOYEN de toutes les équipes
    # (un promu n'a pas d'avantage domicile connu, donc on met la moyenne)
    avantage_saison[numero] = np.mean(list(avantage.values()))   # <-- AJOUTÉ
    index_saison[nom] = numero
    numero += 1
    print(f"{nom} -> attaque {att:.3f}, défense {defe:.3f}")

print(f"\nNombre d'équipes pour la nouvelle saison : {len(index_saison)}")  # doit afficher 20

# ----------------------------------------------------------------------
# 4. SIMULATION DE LA NOUVELLE SAISON AVEC LES 20 ÉQUIPES
# ----------------------------------------------------------------------
from simulateur import Simulateur

# On passe maintenant avantage_saison (bons numéros) au lieu de avantage
sim_nouvelle = Simulateur(forces_saison, avantage_saison, index_saison)   # <-- CORRIGÉ
resultats_nouvelle = sim_nouvelle.simuler_monte_carlo(n_simulations=500)

print("\n--- Classement prédit de la nouvelle saison ---")
print(resultats_nouvelle.to_string())

# ----------------------------------------------------------------------
# 5bis. GRAPHIQUES DE LA NOUVELLE SAISON
# ----------------------------------------------------------------------
# On réutilise ta classe Visualiseur, mais avec les résultats de la nouvelle saison.
visu_nouvelle = Visualiseur(resultats_nouvelle)
visu_nouvelle.graphique_probabilites_titre()
visu_nouvelle.graphique_classement_moyen()
visu_nouvelle.graphique_distribution_points("Arsenal", simulateur=sim_nouvelle)

plt.show()
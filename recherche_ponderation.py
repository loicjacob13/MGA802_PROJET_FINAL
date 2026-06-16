"""
recherche_ponderation.py — Trouve le meilleur trio de pondération des saisons.
Auteurs : Fabien - Loïc - Guillaume — Projet MGA802 Groupe 2

PRINCIPE :
On teste plusieurs trios de poids (ex : 60%-30%-10%) pour pondérer les 3 anciennes
saisons. Pour chaque trio, on entraîne le modèle, on simule la saison 2025/2026, puis
on compare le classement prédit au VRAI classement 2025/2026. Le meilleur trio est
celui dont le classement prédit ressemble le plus au vrai classement.
"""
import numpy as np
import pandas as pd
from donnees import ChargeurDonnees
from modele import ModelePoisson
from simulateur import Simulateur


# ----------------------------------------------------------------------
# 1. CHARGEMENT DES DONNÉES (avec tes fonctions ChargeurDonnees et nettoyer)
# ----------------------------------------------------------------------
c_2223 = ChargeurDonnees("2022-2023.csv")   # la plus ancienne
c_2324 = ChargeurDonnees("2023-2024.csv")   # intermédiaire
c_2425 = ChargeurDonnees("2024-2025.csv")   # la plus récente avant la saison testée
c_2526 = ChargeurDonnees("2025-2026.csv")   # la saison à PRÉDIRE (vrais résultats connus)

c_2223.nettoyer()
c_2324.nettoyer()
c_2425.nettoyer()
c_2526.nettoyer()


# ----------------------------------------------------------------------
# 2. INDEX COMMUN À TOUTES LES SAISONS (correction du bug d'index)
# ----------------------------------------------------------------------
# Ton get_index_equipes() numérote les équipes par ordre alphabétique de CHAQUE
# saison séparément. Comme 3 équipes montent/descendent chaque année, un même nom
# peut avoir un numéro différent d'une saison à l'autre. Si on empile les matchs
# sans corriger, les forces se mélangent entre équipes.
# On construit donc UN SEUL index {nom: numéro} valable pour toutes les saisons.

equipes_toutes = set()                          # ensemble (élimine les doublons)
equipes_toutes |= set(c_2223.get_equipes())     # |= ajoute les équipes de cette saison
equipes_toutes |= set(c_2324.get_equipes())
equipes_toutes |= set(c_2425.get_equipes())
equipes_toutes |= set(c_2526.get_equipes())

equipes_triees = sorted(equipes_toutes)                    # tri alphabétique (reproductible)
index_equipes = {nom: i for i, nom in enumerate(equipes_triees)}   # dictionnaire {nom: numéro}
n_equipes = len(index_equipes)
print(f"\nNombre total d'équipes sur les 4 saisons : {n_equipes}")


# ----------------------------------------------------------------------
# 3. RECONSTRUCTION DES MATCHS AVEC L'INDEX COMMUN
# ----------------------------------------------------------------------
# Même logique que ton get_matchs(), mais on remplace les noms par l'index COMMUN
# au lieu de l'index propre à chaque saison.
def matchs_index_commun(chargeur):
    """Reconstruit [idx_dom, idx_ext, buts_dom, buts_ext] avec l'index commun."""
    # on extrait les 4 colonnes utiles depuis tes données nettoyées (comme get_matchs)
    brutes = chargeur.donnees[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].to_numpy()
    lignes = [
        [index_equipes[ligne[0]],   # numéro équipe domicile (index commun)
         index_equipes[ligne[1]],   # numéro équipe extérieur (index commun)
         int(ligne[2]),             # buts domicile
         int(ligne[3])]             # buts extérieur
        for ligne in brutes
    ]
    return np.array(lignes, dtype=int)

matchs_2223 = matchs_index_commun(c_2223)
matchs_2324 = matchs_index_commun(c_2324)
matchs_2425 = matchs_index_commun(c_2425)
matchs_2526 = matchs_index_commun(c_2526)


# ----------------------------------------------------------------------
# 4. LE VRAI CLASSEMENT 2025/2026 (notre référence à imiter)
# ----------------------------------------------------------------------
# On reprend exactement ta méthode de simuler_saison() : on compte les points
# avec la règle 3/1/0, puis on trie avec pandas .sort_values() (vu en cours).
def vrai_classement(matchs):
    """Calcule le vrai classement final à partir des vrais matchs d'une saison."""
    points = np.zeros(n_equipes, dtype=int)   # points cumulés par équipe

    for idx_dom, idx_ext, buts_dom, buts_ext in matchs:   # parcours de chaque vrai match
        if buts_dom > buts_ext:        # victoire domicile
            points[idx_dom] += 3
        elif buts_dom == buts_ext:     # match nul
            points[idx_dom] += 1
            points[idx_ext] += 1
        else:                          # victoire extérieur
            points[idx_ext] += 3

    # On construit un DataFrame, comme dans ton simuler_saison()
    classement = pd.DataFrame({
        'equipe': list(range(n_equipes)),   # numéro de chaque équipe
        'points': points,
    })
    # Tri par points décroissants (même méthode que ton simuler_saison)
    classement = classement.sort_values('points', ascending=False).reset_index(drop=True)
    classement['position'] = classement.index + 1   # position de 1 à n

    # On range les positions dans un tableau indexé par numéro d'équipe
    positions = np.zeros(n_equipes, dtype=int)
    for _, ligne in classement.iterrows():
        positions[ligne['equipe']] = ligne['position']
    return positions

classement_reel = vrai_classement(matchs_2526)   # référence calculée une seule fois

# On garde seulement les équipes réellement présentes en 2025/2026 pour comparer
equipes_2526 = set(index_equipes[nom] for nom in c_2526.get_equipes())


# ----------------------------------------------------------------------
# 5. ÉVALUATION D'UN TRIO DE PONDÉRATION
# ----------------------------------------------------------------------
def evaluer_trio(p1, p2, p3):
    """
    Entraîne le modèle avec ces poids, simule 2025/2026 et renvoie l'erreur de
    classement. p1=poids 2022-2023, p2=2023-2024, p3=2024-2025.
    Plus l'erreur est petite, meilleur est le trio.
    """
    matchs_tous = np.vstack([matchs_2223, matchs_2324, matchs_2425])   # on empile les 3 saisons

    # un poids par match selon sa saison (comme dans ton main.py)
    poids = np.concatenate([
        np.ones(len(matchs_2223)) * p1,
        np.ones(len(matchs_2324)) * p2,
        np.ones(len(matchs_2425)) * p3,
    ])

    # entraînement avec tes fonctions
    modele = ModelePoisson(n_equipes)
    modele.entrainer(matchs_tous, poids=poids)

    # simulation avec tes fonctions
    forces = modele.get_forces()
    avantage = modele.get_avantage_domicile()
    sim = Simulateur(forces, avantage, index_equipes)
    resultats = sim.simuler_monte_carlo(n_simulations=200)   # 200 simulations suffisent ici

    # on transforme la position moyenne (déjà triée) en classement prédit 1 à n
    classement_predit = np.zeros(n_equipes, dtype=int)
    for position, nom in enumerate(resultats.index):
        classement_predit[index_equipes[nom]] = position + 1

    # erreur = somme des écarts de position, seulement pour les équipes de 2025/2026
    erreur = 0
    for idx in equipes_2526:
        erreur += abs(classement_predit[idx] - classement_reel[idx])
    return erreur


# ----------------------------------------------------------------------
# 6. BOUCLE QUI TESTE TOUS LES TRIOS (la partie recherche)
# ----------------------------------------------------------------------
meilleur_erreur = np.inf   # on minimise, donc on part de l'infini
meilleur_trio = None

pas = np.arange(0.0, 1.01, 0.1)   # poids possibles : 0.0, 0.1, ..., 1.0

for p1 in pas:                     # poids saison la plus ancienne
    for p2 in pas:                 # poids saison intermédiaire
        p3 = 1.0 - p1 - p2         # le 3e poids complète à 100%

        if p3 < 0:                 # trio invalide (poids négatif)
            continue
        if not (p3 >= p2 >= p1):   # on garde "plus récent = plus important"
            continue

        erreur = evaluer_trio(p1, p2, p3)
        print(f"Trio ({p1:.1f}, {p2:.1f}, {p3:.1f}) -> erreur = {erreur}")

        if erreur < meilleur_erreur:   # on garde le meilleur trouvé
            meilleur_erreur = erreur
            meilleur_trio = (p1, p2, p3)


# ----------------------------------------------------------------------
# 7. RÉSULTAT FINAL
# ----------------------------------------------------------------------
print("\n========================================")
print(f"MEILLEUR TRIO TROUVÉ : {meilleur_trio}")
print(f"  -> 2022-2023 : {meilleur_trio[0]*100:.0f}%")
print(f"  -> 2023-2024 : {meilleur_trio[1]*100:.0f}%")
print(f"  -> 2024-2025 : {meilleur_trio[2]*100:.0f}%")
print(f"Erreur de classement totale : {meilleur_erreur}")
print("========================================")
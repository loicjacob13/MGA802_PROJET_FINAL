"""
recherche_ponderation.py — Trouve le meilleur trio de pondération des saisons.
Auteurs : Fabien - Loïc - Guillaume — Projet MGA802 Groupe 2

PRINCIPE :
On teste plusieurs trios de poids pour pondérer 3 saisons d'entraînement. Pour chaque
trio, on entraîne le modèle, on simule une saison déjà jouée (validation), puis on
compare le classement prédit au vrai classement. Le meilleur trio est celui dont le
classement prédit ressemble le plus au vrai classement.
"""
import numpy as np
import pandas as pd
from donnees import ChargeurDonnees
from simulation.modele import ModelePoisson
from simulation.simulateur import Simulateur


def construire_index(liste_chargeurs):
    """Construit un dictionnaire {nom: numéro} commun à plusieurs saisons."""
    noms = []
    for chargeur in liste_chargeurs:
        for nom in chargeur.get_equipes():
            if nom not in noms:                # on évite les doublons
                noms.append(nom)
    noms = sorted(noms)
    index = {}
    for i in range(len(noms)):
        index[noms[i]] = i
    return index


def matchs_avec_index(chargeur, index):
    """Reconstruit les matchs [idx_dom, idx_ext, buts_dom, buts_ext] avec un index donné."""
    brutes = chargeur.donnees[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].to_numpy()
    lignes = []
    for ligne in brutes:
        lignes.append([index[ligne[0]], index[ligne[1]], int(ligne[2]), int(ligne[3])])
    return np.array(lignes, dtype=int)


def trouver_meilleur_trio(fichiers_train, fichier_validation, n_simulations=300, pas=0.03):
    """
    Cherche le meilleur trio de poids (p1, p2, p3) pour pondérer 3 saisons.

    - fichiers_train     : liste des 3 CSV d'entraînement [ancien, ..., récent]
    - fichier_validation : le CSV de la saison déjà jouée servant à valider
    Renvoie le trio (p1, p2, p3) qui minimise l'erreur de classement.
    """
    # --- Chargement des 3 saisons d'entraînement + la saison de validation ---
    c1 = ChargeurDonnees(fichiers_train[0]); c1.nettoyer()
    c2 = ChargeurDonnees(fichiers_train[1]); c2.nettoyer()
    c3 = ChargeurDonnees(fichiers_train[2]); c3.nettoyer()
    cv = ChargeurDonnees(fichier_validation); cv.nettoyer()

    # --- Index commun aux 4 saisons (pour ne pas mélanger les équipes) ---
    index_equipes = construire_index([c1, c2, c3, cv])
    n_equipes = len(index_equipes)

    matchs_1 = matchs_avec_index(c1, index_equipes)
    matchs_2 = matchs_avec_index(c2, index_equipes)
    matchs_3 = matchs_avec_index(c3, index_equipes)

    # --- Index réduit aux équipes de la saison de validation ---
    noms_val = cv.get_equipes()
    index_val = {}
    for i in range(len(noms_val)):
        index_val[noms_val[i]] = i
    n_val = len(index_val)

    # --- Vrai classement de la saison de validation ---
    matchs_val = matchs_avec_index(cv, index_val)
    points = np.zeros(n_val, dtype=int)
    for ligne in matchs_val:
        idx_dom = ligne[0]; idx_ext = ligne[1]
        buts_dom = ligne[2]; buts_ext = ligne[3]
        if buts_dom > buts_ext:
            points[idx_dom] += 3
        elif buts_dom == buts_ext:
            points[idx_dom] += 1
            points[idx_ext] += 1
        else:
            points[idx_ext] += 3
    classement = pd.DataFrame({'equipe': list(range(n_val)), 'points': points})
    classement = classement.sort_values('points', ascending=False).reset_index(drop=True)
    classement['position'] = classement.index + 1
    classement_reel = np.zeros(n_val, dtype=int)
    for i in range(n_val):
        classement_reel[classement['equipe'][i]] = classement['position'][i]

    # --- Évalue un trio : entraîne, simule, renvoie l'erreur ---
    def evaluer_trio(p1, p2, p3):
        matchs_tous = np.vstack([matchs_1, matchs_2, matchs_3])
        poids_1 = np.ones(len(matchs_1)) * p1
        poids_2 = np.ones(len(matchs_2)) * p2
        poids_3 = np.ones(len(matchs_3)) * p3
        poids = np.concatenate([poids_1, poids_2, poids_3])

        modele = ModelePoisson(n_equipes)
        modele.entrainer(matchs_tous, poids=poids)
        forces_completes = modele.get_forces()
        avantage_complet = modele.get_avantage_domicile()

        # on garde seulement les équipes de la saison de validation (renumérotées)
        forces = {}
        avantage = {}
        for nom in index_val:
            idx_reduit = index_val[nom]
            idx_complet = index_equipes[nom]
            forces[idx_reduit] = forces_completes[idx_complet]
            avantage[idx_reduit] = avantage_complet[idx_complet]

        sim = Simulateur(forces, avantage, index_val)
        resultats = sim.simuler_monte_carlo(n_simulations)

        # position prédite de chaque équipe
        classement_predit = np.zeros(n_val, dtype=int)
        rang = 1
        for nom in resultats.index:
            classement_predit[index_val[nom]] = rang
            rang += 1

        # erreur = somme des écarts de position
        erreur = 0
        for i in range(n_val):
            erreur += abs(classement_predit[i] - classement_reel[i])
        return erreur

    # --- Boucle de recherche sur tous les trios valides ---
    meilleur_erreur = np.inf
    meilleur_trio = None
    # On démarre à 'pas' (et non 0) pour que le poids le plus faible ne soit jamais nul
    valeurs = np.arange(pas, 1.01, pas)

    for p1 in valeurs:
        for p2 in valeurs:
            p3 = 1.0 - p1 - p2
            if p3 <= 0:                         # p3 doit être STRICTEMENT positif
                continue
            if not (p3 >= p2 >= p1):            # plus récent = plus important
                continue
            erreur = evaluer_trio(p1, p2, p3)
            if erreur < meilleur_erreur:
                meilleur_erreur = erreur
                meilleur_trio = (round(float(p1), 2), round(float(p2), 2), round(float(p3), 2))

    print(f"Meilleur trio trouvé : {meilleur_trio} (erreur = {meilleur_erreur})")
    return meilleur_trio


# ----------------------------------------------------------------------
# Test rapide si on lance ce fichier directement
# ----------------------------------------------------------------------
if __name__ == "__main__":
    trio = trouver_meilleur_trio(
        ["2021-2022.csv", "2022-2023.csv", "2023-2024.csv"],
        "2024-2025.csv",
    )
    print("Trio renvoyé :", trio)
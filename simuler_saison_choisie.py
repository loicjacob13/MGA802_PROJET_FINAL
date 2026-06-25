"""
simuler_saison_choisie.py — Choisir une saison et la simuler automatiquement.

PRINCIPE :
L'utilisateur choisit une saison. On cherche d'abord le meilleur trio de pondération
(validé sur la saison juste avant celle choisie), puis le modèle s'entraîne sur les
3 saisons précédentes avec ces poids, approxime les forces des équipes promues, et
simule. Si la saison a déjà été jouée, on compare au classement réel.
"""
import numpy as np
import pandas as pd
from donnees import ChargeurDonnees
from modele import ModelePoisson
from simulateur import Simulateur
from forces_promus import forces_pour_position_cible
from recherche_ponderation import trouver_meilleur_trio


# ----------------------------------------------------------------------
# 1. SAISONS DISPONIBLES
# ----------------------------------------------------------------------
# Pour chaque saison on indique :
#   csv          : le fichier de la saison cible (None si elle n'est pas encore jouée)
#   precedentes  : les 3 saisons d'entraînement (plus ancienne -> plus récente)
#   validation   : la saison "juste avant" servant à calibrer les poids
#   train_valid  : les 3 saisons d'entraînement pour cette validation
SAISONS = {
    "2021-2022": {
        "csv": "2021-2022.csv",
        "precedentes": ["2018-2019.csv", "2019-2020.csv", "2020-2021.csv"],
        "validation": "2020-2021.csv",
        "train_valid": ["2017-2018.csv", "2018-2019.csv", "2019-2020.csv"],
    },
    "2022-2023": {
        "csv": "2022-2023.csv",
        "precedentes": ["2019-2020.csv", "2020-2021.csv", "2021-2022.csv"],
        "validation": "2021-2022.csv",
        "train_valid": ["2018-2019.csv", "2019-2020.csv", "2020-2021.csv"],
    },
    "2023-2024": {
        "csv": "2023-2024.csv",
        "precedentes": ["2020-2021.csv", "2021-2022.csv", "2022-2023.csv"],
        "validation": "2022-2023.csv",
        "train_valid": ["2019-2020.csv", "2020-2021.csv", "2021-2022.csv"],
    },
    "2024-2025": {
        "csv": "2024-2025.csv",
        "precedentes": ["2021-2022.csv", "2022-2023.csv", "2023-2024.csv"],
        "validation": "2023-2024.csv",
        "train_valid": ["2020-2021.csv", "2021-2022.csv", "2022-2023.csv"],
    },
    "2025-2026": {
        "csv": "2025-2026.csv",
        "precedentes": ["2022-2023.csv", "2023-2024.csv", "2024-2025.csv"],
        "validation": "2024-2025.csv",
        "train_valid": ["2021-2022.csv", "2022-2023.csv", "2023-2024.csv"],
    },
    "2026-2027": {
        "csv": None,    # saison future : pas encore jouée
        "precedentes": ["2023-2024.csv", "2024-2025.csv", "2025-2026.csv"],
        "validation": "2025-2026.csv",
        "train_valid": ["2022-2023.csv", "2023-2024.csv", "2024-2025.csv"],
        "promus_futurs": ["Coventry", "Ipswich", "Millwall"],
        "relegues_futurs": ["West Ham", "Burnley", "Wolves"],
    },
}

# Positions cibles des 3 promus, identiques pour chaque saison
POSITIONS_PROMUS = [15.0, 16.9, 17.4]


# ----------------------------------------------------------------------
# 2. FONCTIONS UTILITAIRES
# ----------------------------------------------------------------------
def matchs_avec_index(chargeur, index):
    """
    Reconstruit la liste des matchs [idx_dom, idx_ext, buts_dom, buts_ext]
    en remplaçant les noms d'équipes par leur numéro dans 'index'.
    """
    brutes = chargeur.donnees[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].to_numpy()
    lignes = []
    for ligne in brutes:                       # on parcourt chaque match
        nom_dom = ligne[0]
        nom_ext = ligne[1]
        buts_dom = int(ligne[2])
        buts_ext = int(ligne[3])
        lignes.append([index[nom_dom], index[nom_ext], buts_dom, buts_ext])
    return np.array(lignes, dtype=int)


def classement_reel(chargeur, index, n):
    """Calcule le vrai classement final (règle 3 points victoire / 1 nul / 0 défaite)."""
    matchs = matchs_avec_index(chargeur, index)
    points = np.zeros(n, dtype=int)

    for ligne in matchs:                       # on parcourt chaque match
        idx_dom = ligne[0]
        idx_ext = ligne[1]
        buts_dom = ligne[2]
        buts_ext = ligne[3]
        if buts_dom > buts_ext:                # victoire domicile
            points[idx_dom] += 3
        elif buts_dom == buts_ext:             # match nul
            points[idx_dom] += 1
            points[idx_ext] += 1
        else:                                  # victoire extérieur
            points[idx_ext] += 3

    # On trie par points décroissants (méthode pandas vue en cours)
    classement = pd.DataFrame({'equipe': list(range(n)), 'points': points})
    classement = classement.sort_values('points', ascending=False).reset_index(drop=True)
    classement['position'] = classement.index + 1

    # On range la position de chaque équipe dans un tableau
    positions = np.zeros(n, dtype=int)
    for i in range(n):
        numero_equipe = classement['equipe'][i]
        positions[numero_equipe] = classement['position'][i]
    return positions


def construire_index(liste_chargeurs):
    """
    Construit un dictionnaire {nom: numéro} commun à plusieurs saisons.
    On parcourt toutes les équipes et on les numérote par ordre alphabétique.
    """
    noms = []
    for chargeur in liste_chargeurs:
        for nom in chargeur.get_equipes():
            if nom not in noms:                # on évite les doublons
                noms.append(nom)
    noms = sorted(noms)                         # ordre alphabétique (reproductible)
    index = {}
    for i in range(len(noms)):
        index[noms[i]] = i
    return index


# ----------------------------------------------------------------------
# 3. FONCTION PRINCIPALE : SIMULER UNE SAISON
# ----------------------------------------------------------------------
def simuler_saison(nom_saison, n_simulations=500):
    """Calibre les poids, entraîne, approxime les promus, simule et compare au réel."""
    info = SAISONS[nom_saison]
    fichiers_train = info["precedentes"]

    # --- ÉTAPE 0 : trouver le meilleur trio de poids ---
    print("Recherche du meilleur trio de pondération...")
    trio = trouver_meilleur_trio(info["train_valid"], info["validation"])
    p1 = trio[0]
    p2 = trio[1]
    p3 = trio[2]
    print(f"Trio utilisé : {trio}\n")

    # --- ÉTAPE A : entraînement sur les 3 saisons précédentes ---
    c1 = ChargeurDonnees(fichiers_train[0]); c1.nettoyer()
    c2 = ChargeurDonnees(fichiers_train[1]); c2.nettoyer()
    c3 = ChargeurDonnees(fichiers_train[2]); c3.nettoyer()

    # index commun aux 3 saisons d'entraînement
    index_train = construire_index([c1, c2, c3])
    n_train = len(index_train)

    matchs_1 = matchs_avec_index(c1, index_train)
    matchs_2 = matchs_avec_index(c2, index_train)
    matchs_3 = matchs_avec_index(c3, index_train)
    matchs_tous = np.vstack([matchs_1, matchs_2, matchs_3])

    # un poids par match selon sa saison
    poids_1 = np.ones(len(matchs_1)) * p1
    poids_2 = np.ones(len(matchs_2)) * p2
    poids_3 = np.ones(len(matchs_3)) * p3
    poids = np.concatenate([poids_1, poids_2, poids_3])

    modele = ModelePoisson(n_train)
    modele.entrainer(matchs_tous, poids=poids)
    forces_train = modele.get_forces()
    avantage_train = modele.get_avantage_domicile()

    # --- ÉTAPE B : simulation de base pour calibrer les promus ---
    sim_base = Simulateur(forces_train, avantage_train, index_train)
    resultats_base = sim_base.simuler_monte_carlo(n_simulations)

    # --- ÉTAPE C : déterminer les équipes de la saison cible ---
    if info["csv"] is not None:
        # saison déjà jouée : on lit son CSV
        c_cible = ChargeurDonnees(info["csv"]); c_cible.nettoyer()
        equipes_cible = sorted(c_cible.get_equipes())
        # un promu est une équipe de la saison cible absente de l'entraînement
        promus = []
        for nom in equipes_cible:
            if nom not in index_train:
                promus.append(nom)
    else:
        # saison future : équipes récentes - relégués + promus connus
        c_recente = ChargeurDonnees(fichiers_train[2]); c_recente.nettoyer()
        relegues = info["relegues_futurs"]
        promus = info["promus_futurs"]
        equipes_cible = []
        for nom in c_recente.get_equipes():
            if nom not in relegues:
                equipes_cible.append(nom)
        equipes_cible = sorted(equipes_cible + promus)

    print(f"Saison {nom_saison} -- {len(equipes_cible)} équipes")
    print(f"Promus détectés : {promus}")

    # --- ÉTAPE D : construire forces / avantage / index de la saison cible ---
    forces_saison = {}
    avantage_saison = {}
    index_saison = {}
    numero = 0
    for nom in equipes_cible:
        if nom in index_train:
            # équipe connue : on copie ses forces apprises
            idx_t = index_train[nom]
            forces_saison[numero] = forces_train[idx_t]
            avantage_saison[numero] = avantage_train[idx_t]
        else:
            # équipe promue : on approxime via sa position cible
            rang_promu = promus.index(nom)             # 0, 1 ou 2
            if rang_promu > 2:
                rang_promu = 2
            cible = POSITIONS_PROMUS[rang_promu]
            att, defe = forces_pour_position_cible(cible, resultats_base, forces_train, index_train)
            forces_saison[numero] = (att, defe)
            # un promu n'a pas d'avantage domicile connu -> on met la moyenne des autres
            avantage_saison[numero] = np.mean(list(avantage_train.values()))
        index_saison[nom] = numero
        numero += 1

    # --- ÉTAPE E : simulation finale de la saison cible ---
    sim = Simulateur(forces_saison, avantage_saison, index_saison)
    resultats = sim.simuler_monte_carlo(n_simulations)

    print("\n--- CLASSEMENT SIMULÉ ---")
    print(resultats.to_string())

    # --- ÉTAPE F : comparaison au réel si la saison a déjà été jouée ---
    if info["csv"] is not None:
        n_cible = len(index_saison)
        positions_reelles = classement_reel(c_cible, index_saison, n_cible)

        # position simulée de chaque équipe (resultats est trié par position moyenne)
        position_simulee = {}
        rang = 1
        for nom in resultats.index:
            position_simulee[nom] = rang
            rang += 1

        print("\n--- COMPARAISON SIMULÉ vs RÉEL ---")
        print(f"{'Équipe':<20}{'Simulé':>8}{'Réel':>8}{'Écart':>8}")
        print("-" * 44)
        # on affiche dans l'ordre du classement réel : on parcourt les positions 1 à n
        for position in range(1, n_cible + 1):
            for nom in index_saison:                   # on cherche l'équipe à cette position
                if positions_reelles[index_saison[nom]] == position:
                    pos_sim = position_simulee[nom]
                    pos_reel = position
                    ecart = abs(pos_sim - pos_reel)
                    print(f"{nom:<20}{pos_sim:>8}{pos_reel:>8}{ecart:>8}")
    else:
        print("\n(Saison future : pas de classement réel pour comparer.)")

    return resultats


# ----------------------------------------------------------------------
# 4. PROGRAMME PRINCIPAL : DEMANDE À L'UTILISATEUR
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Saisons disponibles :")
    for nom in SAISONS:
        if SAISONS[nom]["csv"] is None:
            print(f"  - {nom} (future)")
        else:
            print(f"  - {nom} (déjà jouée)")

    choix = input("\nQuelle saison voulez-vous simuler ? (ex: 2023-2024) : ").strip()
    while choix not in SAISONS:
        print("Saison invalide. Choisissez parmi la liste ci-dessus.")
        choix = input("Quelle saison ? : ").strip()

    simuler_saison(choix, n_simulations=500)
"""
main_NEW.py — Choisir une saison et la simuler automatiquement.
Auteurs : Fabien - Loïc - Guillaume — Projet MGA802 Groupe 2

PRINCIPE :
L'utilisateur choisit une saison. On cherche d'abord le meilleur trio de pondération
(validé sur la saison juste avant celle choisie), puis le modèle s'entraîne sur les
3 saisons précédentes avec ces poids, approxime les forces des équipes promues, et
simule. Si la saison a déjà été jouée, on compare au classement réel.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from donnees import ChargeurDonnees
from simulation.modele import ModelePoisson
from simulation.simulateur import Simulateur
from visualisation import Visualiseur
from simulation.forces_promus import forces_pour_position_cible
from simulation.recherche_ponderation import trouver_meilleur_trio
from controle_input import demander_saison, demander_entier_positif, demander_sigle


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
        "promus": ["Norwich", "Watford", "Brentford"],   # 1er, 2e, 3e promu
    },
    "2022-2023": {
        "csv": "2022-2023.csv",
        "precedentes": ["2019-2020.csv", "2020-2021.csv", "2021-2022.csv"],
        "validation": "2021-2022.csv",
        "train_valid": ["2018-2019.csv", "2019-2020.csv", "2020-2021.csv"],
        "promus": ["Fulham", "Bournemouth", "Nott'm Forest"],
    },
    "2023-2024": {
        "csv": "2023-2024.csv",
        "precedentes": ["2020-2021.csv", "2021-2022.csv", "2022-2023.csv"],
        "validation": "2022-2023.csv",
        "train_valid": ["2019-2020.csv", "2020-2021.csv", "2021-2022.csv"],
        "promus": ["Burnley", "Sheffield United", "Luton"],
    },
    "2024-2025": {
        "csv": "2024-2025.csv",
        "precedentes": ["2021-2022.csv", "2022-2023.csv", "2023-2024.csv"],
        "validation": "2023-2024.csv",
        "train_valid": ["2020-2021.csv", "2021-2022.csv", "2022-2023.csv"],
        "promus": ["Leicester", "Ipswich", "Southampton"],
    },
    "2025-2026": {
        "csv": "2025-2026.csv",
        "precedentes": ["2022-2023.csv", "2023-2024.csv", "2024-2025.csv"],
        "validation": "2024-2025.csv",
        "train_valid": ["2021-2022.csv", "2022-2023.csv", "2023-2024.csv"],
        "promus": ["Leeds", "Burnley", "Sunderland"],
    },
    "2026-2027": {
        "csv": None,    # saison future : pas encore jouée
        "precedentes": ["2023-2024.csv", "2024-2025.csv", "2025-2026.csv"],
        "validation": "2025-2026.csv",
        "train_valid": ["2022-2023.csv", "2023-2024.csv", "2024-2025.csv"],
        "promus": ["Coventry", "Ipswich", "Millwall"],
        "relegues_futurs": ["West Ham", "Burnley", "Wolves"],
    },
}

# Positions cibles des 3 promus, identiques pour chaque saison
POSITIONS_PROMUS = [15.0, 16.9, 17.4]

# Sigle court de chaque équipe (pour faciliter la saisie de l'utilisateur)
SIGLES = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU", "Brentford": "BRE",
    "Brighton": "BHA", "Burnley": "BUR", "Chelsea": "CHE", "Crystal Palace": "CRY",
    "Everton": "EVE", "Fulham": "FUL", "Ipswich": "IPS", "Leeds": "LEE",
    "Leicester": "LEI", "Liverpool": "LIV", "Luton": "LUT", "Man City": "MCI",
    "Man United": "MUN", "Newcastle": "NEW", "Norwich": "NOR", "Nott'm Forest": "NFO",
    "Sheffield United": "SHU", "Southampton": "SOU", "Sunderland": "SUN",
    "Tottenham": "TOT", "Watford": "WAT", "West Ham": "WHU", "Wolves": "WOL",
    "Coventry": "COV", "Millwall": "MIL",
}


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

    # --- ÉTAPE B : déterminer les équipes de la saison cible ---
    # Les promus sont déclarés manuellement (dans l'ordre 1er, 2e, 3e promu).
    promus = info["promus"]

    if info["csv"] is not None:
        # saison déjà jouée : on lit son CSV pour avoir la liste exacte des 20 équipes
        c_cible = ChargeurDonnees(info["csv"]); c_cible.nettoyer()
        equipes_cible = sorted(c_cible.get_equipes())
    else:
        # saison future : équipes récentes - relégués + promus connus
        c_recente = ChargeurDonnees(fichiers_train[2]); c_recente.nettoyer()
        relegues = info["relegues_futurs"]
        equipes_cible = []
        for nom in c_recente.get_equipes():
            if nom not in relegues:
                equipes_cible.append(nom)
        equipes_cible = sorted(equipes_cible + promus)

    print(f"Saison {nom_saison} -- {len(equipes_cible)} équipes")
    print(f"Promus (manuels) : {promus}")

    # --- ÉTAPE C : construire forces / avantage / index de la saison cible ---
    forces_saison = {}
    avantage_saison = {}
    index_saison = {}
    numero = 0
    for nom in equipes_cible:
        # une équipe est traitée comme promue si elle est dans la liste manuelle,
        # MÊME si elle a déjà joué en PL avant (cas des équipes "yo-yo")
        if nom in promus:
            # équipe promue : on approxime via sa position cible.
            # La fonction entraîne le modèle saison par saison (20 équipes propres),
            # trouve l'équipe la plus proche de la cible dans chaque saison, et moyenne.
            rang_promu = promus.index(nom)             # 0, 1 ou 2
            if rang_promu > 2:
                rang_promu = 2
            cible = POSITIONS_PROMUS[rang_promu]
            att, defe, av = forces_pour_position_cible(cible, fichiers_train, n_simulations)
            forces_saison[numero] = (att, defe)
            avantage_saison[numero] = av
        elif nom in index_train:
            # équipe connue (maintenue) : on copie ses forces apprises
            idx_t = index_train[nom]
            forces_saison[numero] = forces_train[idx_t]
            avantage_saison[numero] = avantage_train[idx_t]
        else:
            # cas rare : équipe ni promue ni connue -> on la traite comme un promu moyen
            att, defe, av = forces_pour_position_cible(17.0, fichiers_train, n_simulations)
            forces_saison[numero] = (att, defe)
            avantage_saison[numero] = av
        index_saison[nom] = numero
        numero += 1

    # --- ÉTAPE D : simulation finale de la saison cible ---
    sim = Simulateur(forces_saison, avantage_saison, index_saison)
    resultats = sim.simuler_monte_carlo(n_simulations)

    print("\n--- CLASSEMENT SIMULÉ ---")
    print(resultats.to_string())

    # --- ÉTAPE E : comparaison au réel si la saison a déjà été jouée ---
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

    # on renvoie aussi le simulateur et l'index pour pouvoir tracer les graphiques
    return resultats, sim, index_saison


# ----------------------------------------------------------------------
# 4. PROGRAMME PRINCIPAL
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. On affiche les saisons disponibles
    print("Saisons disponibles :")
    for nom in SAISONS:
        if SAISONS[nom]["csv"] is None:
            print(f"  - {nom} (future)")
        else:
            print(f"  - {nom} (déjà jouée)")

    # 2. On demande une saison valide (fonction du module validation_saisie)
    saison_selectionnee = demander_saison(
        "\nQuelle saison voulez-vous simuler ? (ex: 2023-2024) : ", SAISONS
    )

    # 3. On demande le nombre de simulations (entier positif contrôlé)
    nb_simulations = demander_entier_positif(
        "Combien de simulations voulez-vous effectuer ? (ex: 500) : "
    )

    print(f"\nLancement de la simulation pour {saison_selectionnee} ({nb_simulations} simulations)...")
    # On lance la simulation : elle renvoie les résultats, le simulateur et l'index
    resultats, sim, index_saison = simuler_saison(saison_selectionnee, n_simulations=nb_simulations)

    # ------------------------------------------------------------------
    # GRAPHIQUES (on réutilise la classe Visualiseur)
    # ------------------------------------------------------------------
    # On affiche la liste des équipes avec leur sigle officiel
    print("\nÉquipes de cette saison :")
    for nom in sorted(index_saison.keys()):
        sigle = SIGLES[nom]
        print(f"  {sigle:<5} = {nom}")

    # dictionnaire {sigle: nom} pour retrouver le nom complet depuis le sigle
    sigle_vers_nom = {}
    for nom in index_saison:
        sigle_vers_nom[SIGLES[nom]] = nom

    # 4. On demande un sigle officiel valide (fonction du module validation_saisie)
    equipe = demander_sigle(
        "\nQuelle équipe détailler ? (sigle officiel, ex: ARS) : ", sigle_vers_nom
    )

    visu = Visualiseur(resultats)
    visu.graphique_probabilites_titre()
    visu.graphique_classement_moyen()
    visu.graphique_distribution_points(equipe, simulateur=sim)
    plt.show()


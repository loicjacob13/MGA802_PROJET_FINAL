"""
forces_promus.py — Approxime les forces des équipes promues.
Auteurs : Fabien - Loïc - Guillaume — Projet MGA802 Groupe 2

PRINCIPE :
Une équipe promue n'a pas joué en Premier League, donc le modèle ne connaît pas ses
forces. On les approxime à partir des positions cibles historiques (15.0 pour le 1er
promu, 16.9 pour le 2e, 17.4 pour le 3e).

Pour rester COHÉRENT avec le modèle, on procède saison par saison :
  - pour CHACUNE des 3 saisons d'entraînement, on entraîne le modèle sur cette seule
    saison (20 équipes), on simule un classement, et on note les forces de l'équipe
    la plus proche de la position cible.
  - on moyenne ces forces (et l'avantage domicile) sur les 3 saisons.

Chaque saison a exactement ses 20 équipes : la position cible (ex 15e sur 20) est donc
fiable, sans être faussée par les équipes cumulées de plusieurs saisons.
"""
import numpy as np
from donnees import ChargeurDonnees
from simulation.modele import ModelePoisson
from simulation.simulateur import Simulateur


def equipe_la_plus_proche(resultats, position_cible):
    """Renvoie le NOM de l'équipe dont la position moyenne est la plus proche de la cible."""
    ecarts = (resultats['position_moyenne'] - position_cible).abs()   # écart à la cible
    nom_plus_proche = ecarts.sort_values().index[0]                   # la 1ère = la plus proche
    return nom_plus_proche


def forces_d_une_saison(fichier_csv, n_simulations):
    """
    Entraîne le modèle sur UNE seule saison, simule, et renvoie :
      - resultats   : le classement simulé (DataFrame)
      - forces      : dict {index: (attaque, defense)}
      - avantages   : dict {index: avantage_domicile}
      - index       : dict {nom: index}
    """
    c = ChargeurDonnees(fichier_csv)
    c.nettoyer()
    index = c.get_index_equipes()
    matchs = c.get_matchs()

    modele = ModelePoisson(len(index))
    modele.entrainer(matchs)                       # une seule saison : pas de pondération
    forces = modele.get_forces()
    avantages = modele.get_avantage_domicile()

    sim = Simulateur(forces, avantages, index)
    resultats = sim.simuler_monte_carlo(n_simulations)
    return resultats, forces, avantages, index


def forces_pour_position_cible(position_cible, fichiers_saisons, n_simulations=300):
    """
    Approxime (attaque, defense, avantage) d'un promu pour une position cible donnée.

    - position_cible  : la position visée (ex 15.0, 16.9, 17.4)
    - fichiers_saisons: liste des 3 CSV des saisons d'entraînement
    - n_simulations   : nombre de simulations Monte-Carlo par saison

    Pour chaque saison : on trouve l'équipe la plus proche de la cible et on prend ses
    forces. On moyenne ensuite sur les 3 saisons.
    """
    attaques = []
    defenses = []
    avantages_promu = []
    equipes_choisies = []

    for fichier in fichiers_saisons:               # on parcourt chaque saison séparément
        resultats, forces, avantages, index = forces_d_une_saison(fichier, n_simulations)

        # l'équipe la plus proche de la position cible dans CETTE saison
        nom = equipe_la_plus_proche(resultats, position_cible)
        idx = index[nom]
        att, defe = forces[idx]

        attaques.append(att)
        defenses.append(defe)
        avantages_promu.append(avantages[idx])
        equipes_choisies.append(nom)

    # moyenne des forces sur les 3 saisons
    attaque_moyenne = np.mean(attaques)
    defense_moyenne = np.mean(defenses)
    avantage_moyen = np.mean(avantages_promu)

    print(f"Position cible {position_cible} -> équipes choisies par saison : {equipes_choisies}")
    return float(attaque_moyenne), float(defense_moyenne), float(avantage_moyen)
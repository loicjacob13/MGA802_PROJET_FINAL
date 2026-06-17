"""
PRINCIPE :
Une équipe promue n'a pas (ou peu) de matchs en Premier League, donc le modèle ne
peut pas apprendre ses forces. On les approxime : on connaît la position moyenne que
finissent historiquement les promus (15.0 pour le 1er promu, 16.9 pour le 2e, 17.4
pour le 3e). On cherche les 3 anciennes équipes dont la position moyenne simulée est
la plus proche de cette cible, et on copie la MOYENNE de leurs forces.
"""
import numpy as np


def forces_pour_position_cible(position_cible, resultats, forces, index_equipes):
    """
    Renvoie une force (attaque, defense) approximée pour une position cible donnée.

    - position_cible : la position moyenne visée (ex 15.0, 16.9, 17.4)
    - resultats      : le DataFrame renvoyé par simuler_monte_carlo (contient 'position_moyenne')
    - forces         : le dict {index: (attaque, defense)} renvoyé par get_forces()
    - index_equipes  : le dict {nom: index}
    """
    # On calcule, pour chaque équipe, l'écart entre sa position moyenne et la cible
    ecarts = (resultats['position_moyenne'] - position_cible).abs()   # .abs() = valeur absolue

    # On trie par écart croissant et on garde les 3 équipes les plus proches
    trois_plus_proches = ecarts.sort_values().head(3).index   # head(3) = les 3 premières

    # On récupère les forces (attaque, defense) de ces 3 équipes
    attaques = []
    defenses = []
    for nom in trois_plus_proches:
        idx = index_equipes[nom]          # numéro de l'équipe
        att, defe = forces[idx]           # ses forces apprises
        attaques.append(att)
        defenses.append(defe)

    # On fait la moyenne des 3 attaques et des 3 défenses
    attaque_moyenne = np.mean(attaques)
    defense_moyenne = np.mean(defenses)

    print(f"Position cible {position_cible} -> moyenne de : {list(trois_plus_proches)}")
    return float(attaque_moyenne), float(defense_moyenne)
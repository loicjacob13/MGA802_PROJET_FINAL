"""
mesure_temps.py — Mesure du temps d'exécution (tic / toc).
Réutilise le principe perf_counter du Mini-Projet B.
"""
from time import perf_counter


def chrono_simulation(fonction, nom_saison, n_simulations):
    """
    Mesure le temps d'exécution d'une simulation de saison.

    Args:
        fonction: la fonction de simulation à mesurer (simuler_saison).
        nom_saison (str): la saison à simuler (ex: "2023-2024").
        n_simulations (int): nombre de simulations Monte-Carlo (choisi par l'user).

    Returns:
        tuple: (resultats, sim, index_saison, duree) où duree est en secondes.
    """
    tic = perf_counter()                       # on démarre le chrono
    resultats, sim, index_saison = fonction(nom_saison, n_simulations)
    toc = perf_counter()                       # on arrête le chrono

    duree = toc - tic
    print(f"\n[TEMPS] Simulation de {nom_saison} : {duree:.6f} [s]")
    return resultats, sim, index_saison, duree
"""
validation_saisie.py — Fonctions de saisie contrôlée pour l'utilisateur.

PRINCIPE :
On regroupe ici les fonctions qui demandent une saisie à l'utilisateur et qui
vérifient qu'elle est valide. Tant que la saisie est incorrecte, on redemande
(boucle while True + try/except), ce qui évite que le programme plante.
"""


def demander_saison(message, saisons_valides):
    """
    Demande une saison à l'utilisateur et renvoie son nom.
    Tant que la saisie ne correspond pas à une saison valide, on redemande.
    Gère les espaces en trop et les fautes de frappe (ex: "2103-2023").

    - message         : le texte affiché à l'utilisateur
    - saisons_valides : le dictionnaire (ou la liste) des saisons autorisées
    """
    while True:
        saisie = input(message).strip()        # .strip() enlève les espaces avant/après
        if saisie in saisons_valides:          # la saisie correspond à une vraie saison
            return saisie
        print("Erreur : saison invalide. Choisissez une saison de la liste (ex : 2023-2024).")


def demander_entier_positif(message):
    """
    Demande un nombre entier strictement positif (ex : nombre de simulations).
    Tant que la saisie n'est pas un entier positif, on redemande.
    """
    while True:
        try:
            valeur = int(input(message))       # échoue si ce n'est pas un entier
        except ValueError:
            print("Erreur : saisissez un nombre entier (ex : 500).")
            continue                           # on redemande
        if valeur <= 0:
            print("Erreur : le nombre de simulations doit être strictement positif.")
            continue                           # on redemande
        return valeur


def demander_sigle(message, sigle_vers_nom):
    """
    Demande un sigle officiel et renvoie le NOM complet de l'équipe.
    Tant que le sigle n'existe pas dans la saison, on redemande.
    Gère les espaces, la casse (ars/ARS) et les sigles faux (ARSS, AR, "").

    - message        : le texte affiché à l'utilisateur
    - sigle_vers_nom : dictionnaire {sigle: nom} des équipes de la saison
    """
    while True:
        saisie = input(message).strip().upper()    # .strip() enlève les espaces, .upper() met en majuscules
        if saisie in sigle_vers_nom:                # le sigle existe dans cette saison
            return sigle_vers_nom[saisie]
        print("Erreur : sigle invalide. Entrez un sigle de la liste ci-dessus (ex : ARS).")
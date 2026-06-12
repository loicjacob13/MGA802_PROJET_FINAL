# exploration_donnees.py est le fichier pour l'exploration dans la base de données complète de la Premier League
# Auteurs : Fabien — Loïc - Guillaume - Projet MGA802 Groupe 2
# Ce script produit ne grille MultiIndex par saison (visualisation de tous les matchs)
# ainsi que les statistiques consolidées par équipe et un résumé des tableaux NumPy
# que get_matchs() fournit au modèle

import pandas as pd
import numpy as np
from donnees import ChargeurDonnees #on importe la classe qu'on a codée qui est dans le même dossier

SOUS_COLONNES = ["journee", "buts_dom", "buts_ext", "resultat",     #sous-colonnes affichées dans chaque case de la grille
                 "pts_dom", "pts_ext", "diff_buts", "total_buts"]

FICHIERS = {                         #les 3 fichiers CSV à charger
    "2023-2024": "2023-2024.csv",
    "2024-2025": "2024-2025.csv",
    "2025-2026": "2025-2026.csv",
}

def construire_grille(chargeur):
    """
    Construit la grille MultiIndex à partir d'un ChargeurDonnees déjà nettoyé
    En colonnes : MultiIndex (equipe_domicile, sous_colonne)
    Pour les lignes : equipe_exterieure
    Chaque case (ext, dom) décrit le match joué entre ces deux équipes
    La diagonale reste vide (une équipe ne joue pas contre elle-même)
    Les paramètres sont chargeur (ChargeurDonnees) : instance déjà chargée et nettoyée
    La fonciton retourne un pandas.DataFrame avec MultiIndex en colonnes
    """
    equipes = chargeur.get_equipes()     #liste triée des 20 équipes de la saison
    index_eq = chargeur.get_index_equipes()   #{nom: numéro} pour retrouver l'index

    colonnes = pd.MultiIndex.from_product([equipes, SOUS_COLONNES],      #pd.MultiIndex.from_product crée toutes les combinaisons (equipe × sous-colonne)
                                          names=["Domicile", "info"])
    grille = pd.DataFrame(index=equipes, columns=colonnes, dtype=object)       #DataFrame vide : lignes = équipes extérieures & colonnes = MultiIndex
    grille.index.name = "Exterieur"

    matchs_numpy = chargeur.get_matchs()      #on récupère le tableau NumPy : [idx_dom, idx_ext, buts_dom, buts_ext]

    df = chargeur.donnees.sort_values('Date').reset_index(drop=True)  #besoin des dates pour calculer les journées, on va trier le DataFrame interne par date pour numéroter les journées
    df['journee'] = df.index // 10 + 1   #numéro de journée, on a 10 matchs par journée en Premier League

    noms = {v: k for k, v in index_eq.items()}     #dictionnaire inverse {numéro: nom} pour retrouver les noms depuis les index NumPy


    for i, ligne in df.iterrows():   #on parcourt chaque match du DataFrame trié
        dom = ligne['HomeTeam'] #nom de l'équipe à domicile
        ext = ligne['AwayTeam']    #nom de l'équipe à l'extérieur
        bd = ligne['FTHG']     #buts domicile
        be = ligne['FTAG']    #buts extérieur
        j = ligne['journee']     #numéro de journée calculé

        if bd > be:        #résultat du point de vue de l'équipe à domicile
            res, pts_dom, pts_ext = "V", 3, 0  #victoire à domicile
        elif bd == be:
            res, pts_dom, pts_ext = "N", 1, 1   #match nul
        else:
            res, pts_dom, pts_ext = "D", 0, 3     #défaite à domicile

        valeurs = {                    #on remplit la case (ext, dom) de la grille
            "journee": j,
            "buts_dom": bd,
            "buts_ext": be,
            "resultat": res,
            "pts_dom": pts_dom,
            "pts_ext": pts_ext,
            "diff_buts": bd - be,   #différence de buts
            "total_buts": bd + be,     #total de buts dans le match
        }

        for col, val in valeurs.items():
            grille.loc[ext, (dom, col)] = val   #.loc[ligne, (col_niv1, col_niv2)]

    return grille


def afficher_stats_equipes(chargeur, saison):
    """
    Cette fonction affiche les statistiques de buts moyens par équipe pour une
    saison donnée. Ce sont les données que le modèle utilisera pour s'initialiser.
    Ses paramètres sont chargeur (ChargeurDonnees) : instance déjà nettoyée
       &  saison (str) : label de la saison, ex. "2023-2024"
    """
    stats = chargeur.statistiques_equipes()    #DataFrame buts_marques_moy / buts_encaisses_moy

    print(f"\nStatistiques par équipe à la saison {saison}")
    print(stats.round(3).to_string())

def afficher_resume_numpy(chargeur, saison):
    """
    Affiche un aperçu du tableau NumPy que get_matchs() fournit au modèle après
    Le format est le suivant [index_dom, index_ext, buts_dom, buts_ext]
    les paramètres sont chargeur & saison
    """
    matchs = chargeur.get_matchs()     #tableau NumPy (380, 4)
    index = chargeur.get_index_equipes()

    noms = {v: k for k, v in index.items()}   #dictionnaire inverse pour afficher les noms à côté des index

    print(f"\nTableau NumPy get_matchs() à la saison {saison}")
    print(f"Shape : {matchs.shape}  avec le format : [idx_dom, idx_ext, buts_dom, buts_ext]")
    print(f"5 premières lignes :")

    for ligne in matchs[:5]:    #on affiche les 5 premières lignes
        idx_d, idx_e, bd, be = ligne
        print(f"  [{idx_d:2d}, {idx_e:2d}, {bd}, {be}]"
              f"  →  {noms[idx_d]:<20} vs {noms[idx_e]:<20}  {bd}-{be}")

if __name__ == "__main__":

    pd.set_option("display.max_columns", None)      #options d'affichage Pandas pour voir toutes les colonnes

    pd.set_option("display.width", 300)
    pd.set_option("display.max_rows", None)

    chargeurs = {}    #on stocke les chargeurs pour la consolidation finale

    for saison, fichier in FICHIERS.items():
        print("\n" + "=" * 80)
        print(f"  SAISON {saison}")
        print("=" * 80)

        c = ChargeurDonnees(fichier)        #chargement et nettoyage via la classe ChargeurDonnees

        c.nettoyer()
        chargeurs[saison] = c    #on garde l'instance pour la consolidation

        print(f"\n  {len(c.get_equipes())} équipes : {c.get_equipes()}")
        print(f"  Correspondance nom --> index : {c.get_index_equipes()}")
        print(f"\n--- Grille complète des matchs (saison {saison}) ---\n")

        grille = construire_grille(c)    #grille MultiIndex complète de la saison
        print(grille.fillna("---"))         #.fillna("---") remplace les cases vides par "---"

        n = grille.xs("journee", axis=1, level="info").notna().sum().sum()          #nombre de matchs vérifiés dans la grille

        print(f"\nMatchs dans la grille : {n} / {len(c.get_matchs())}")

        afficher_stats_equipes(c, saison)          #statistiques par équipe
        afficher_resume_numpy(c, saison)         #aperçu du tableau NumPy


    #Par la suite, les données vont être consolidées sur 3 saisons
    # Ce sera utile pour entraîner le modèle sur plusieurs saisons à la fois

    print("\n" + "=" * 80)
    print("  DONNÉES CONSOLIDÉES SUR LES 3 SAISONS COMBINÉES")
    print("=" * 80)

    c_tout = ChargeurDonnees(list(FICHIERS.values()))    #on charge les 3 saisons d'un coup avec la liste de chemins

    c_tout.nettoyer()

    matchs_tout = c_tout.get_matchs()    #tableau NumPy (1140, 4)
    print(f"\nTableau NumPy consolidé : {matchs_tout.shape}")
    print(f"  ({len(c_tout.get_equipes())} équipes uniques sur 3 saisons)")
    print(f"  Équipes : {c_tout.get_equipes()}")

    print(f"\nStatistiques consolidées par équipe (moyennes sur 3 saisons) :")     #statistiques consolidées moyennes sur les 3 saisons

    stats_tout = c_tout.statistiques_equipes()
    print(stats_tout.round(3).to_string())

    c_tout.sauvegarder("matchs_consolides.pkl")       #sauvegarde des données consolidées en pickle

    #
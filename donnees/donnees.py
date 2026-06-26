"""donnees.py — Ce fichier permet de charger et nettoyer les données de Premier League.
Auteurs : Fabien - Loïc - Guillaume — Projet MGA802 Groupe 2
"""

import pandas as pd
import numpy as np

class ChargeurDonnees:   #création de la classe
    """
    Charge, nettoie et donne accès aux données de matchs de Premier League (3 CSV).

    Accepte un seul chemin CSV (str) ou une liste de chemins (list) si on veut
    fusionner plusieurs saisons.
    """

    def __init__(self, chemin_csv):
        """
        Charge le ou les CSV et stocke le résultat dans self.donnees.

        :param chemin_csv: chemin d'un CSV, ou liste de chemins de CSV à fusionner.
        :type chemin_csv: str | list
        :raises FileNotFoundError: si un fichier CSV est introuvable.
        :raises TypeError: si chemin_csv n'est ni une str ni une list.
        """
        ENCODAGE = 'utf-8-sig'  #encoding utf8 gère le BOM (caractère parasite présent dans certains CSV) — si aucuns parasites, il ne fait rien

        if isinstance(chemin_csv, str):            #on vérifie si l'argument reçu est une simple chaîne de caractères (un seul fichier)

            try:
                self.donnees = pd.read_csv('../donnees/' + chemin_csv, encoding=ENCODAGE) #lecture du csv dans un DataFrame
                print(f"Fichier chargé : {chemin_csv} ({len(self.donnees)} matchs)")
            except FileNotFoundError:
                raise FileNotFoundError(                #on lève une erreur si le fichier n'existe pas
                    f"Fichier introuvable : '{chemin_csv}'. "
                    f"Veuillez vérifier que le fichier CSV est bien dans le bon dossier"
                )

        elif isinstance(chemin_csv, list):    #si l'argument est une liste, on charge chaque fichier et on les empile
            morceaux = []     #liste vide qui va accueillir chaque DataFrame chargé

            for chemin in chemin_csv:  #on parcourt chaque chemin de la liste
                try:
                    df_temp = pd.read_csv(chemin, encoding=ENCODAGE)     #lecture d'un fichier .csv
                    print(f"Fichier chargé : {chemin} ({len(df_temp)} matchs)")
                    morceaux.append(df_temp)      #on ajoute ce DataFrame à la liste
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"Fichier introuvable : '{chemin}'. "
                        f"Vérifiez que le CSV est bien dans le dossier"
                    )

            self.donnees = pd.concat(morceaux, ignore_index=True) #pd.concat empile tous les DataFrames et ignore_index repart à 0 après fusion

            print(f"Total après la fusion : {len(self.donnees)} matchs")

        else:     #si ce n'est ni une str ni une liste, on refuse avec un message clair
            raise TypeError(
                "Le chemin_csv doit être une chaîne (str) ou une liste de chemins (list)"
            )

    def nettoyer(self):
        """
        Prépare les données brutes pour le modèle en filtrant les colonnes
        utiles, en convertissant les dates et en supprimant les lignes vides.

        :return: rien ; modifie self.donnees en place.
        :rtype: None
        """
        colonnes_utiles = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']  #on  enlève les colonnes inutiles
        self.donnees = self.donnees[colonnes_utiles]
        self.donnees['Date'] = pd.to_datetime(self.donnees['Date'], dayfirst=True)  #la colonne Date est lue comme du texte qu'on va convertir en vrai objet date
        nb_avant = len(self.donnees)      #on compte les lignes avant suppression pour informer l'utilisateur
        self.donnees = self.donnees.dropna()     #.dropna() supprime toutes les lignes qui contiennent au moins une valeur vide
        nb_apres = len(self.donnees)
        if nb_avant != nb_apres:     #si des lignes ont été supprimées, on le signale
            print(f"Il y a : {nb_avant - nb_apres} lignes qui on été supprimées")

        self.donnees = self.donnees.reset_index(drop=True)     #on reajuste les numéros de lignes après la suppression
        self.donnees['FTHG'] = self.donnees['FTHG'].astype(int) #on fait lire les clonnes de buts en int et pas float
        self.donnees['FTAG'] = self.donnees['FTAG'].astype(int)
        self.donnees = self.donnees.sort_values('Date').reset_index(drop=True)  # on trie par date pour avoir les matchs dans l'ordre chronologique
        self.donnees['journee'] = self.donnees.index + 1  # numéro de match dans l'ordre chronologique, de 1 à 380
        print(f"Données nettoyées : {len(self.donnees)} matchs conservés.")

    def get_equipes(self):
        """
        Retourne la liste triée alphabétiquement de toutes les équipes uniques.

        :return: liste des noms d'équipes, triée alphabétiquement.
        :rtype: list
        """
        equipes_dom = self.donnees['HomeTeam'].unique()   #équipes qui ont joué à domicile
        equipes_ext = self.donnees['AwayTeam'].unique()    #équipes qui ont joué à l'extérieur
        toutes_equipes = set(equipes_dom) | set(equipes_ext)        #set() élimine les doublons, l'opérateur | fait l'union des deux ensembles

        return sorted(toutes_equipes)     #sorted() trie par ordre alphabétique et renvoie une liste

    def get_index_equipes(self):
        """
        Associe à chaque équipe un numéro entier selon sa position dans la liste triée.

        :return: dictionnaire {nom_equipe: numero_entier}.
        :rtype: dict
        """
        equipes = self.get_equipes()    #liste triée des équipes
        index_equipes = {nom: numero for numero, nom in enumerate(equipes)}     #dict comprehension avec enumerate()
        return index_equipes

    def get_matchs(self):
        """
        Retourne tous les matchs sous forme de tableau d'entiers.

        Chaque ligne suit le format [index_dom, index_ext, buts_dom, buts_ext].

        :return: tableau NumPy de forme (n_matchs, 4) au format entier.
        :rtype: numpy.ndarray
        """
        index_equipes = self.get_index_equipes()  #dictionnaire {nom: numéro}
        donnees_brutes = self.donnees[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].to_numpy() #on extrait les 4 colonnes utiles et on convertit en tableau NumPy
        lignes = [                   #on remplace chaque nom d'équipe par son numéro entier
            [
                index_equipes[ligne[0]],     #numéro de l'équipe à domicile
                index_equipes[ligne[1]],  #numéro de l'équipe à l'extérieur
                int(ligne[2]),              #buts marqués par l'équipe à domicile
                int(ligne[3])               #buts marqués par l'équipe à l'extérieur
            ]
            for ligne in donnees_brutes    #on parcourt chaque match
        ]

        return np.array(lignes, dtype=int)     #np.array convertit la liste en tableau NumPy, dtype=int force le type entier

    def statistiques_equipes(self):
        """
        Calcule les buts moyens marqués et encaissés pour chaque équipe.

        :return: DataFrame indexé par nom d'équipe, colonnes buts_marques_moy
            et buts_encaisses_moy.
        :rtype: pandas.DataFrame
        """

        stats_dom = self.donnees.groupby('HomeTeam').agg(  #.groupby() regroupe les matchs par équipe, .mean() calcule la moyenne
            buts_marques_dom=('FTHG', 'mean'),       #quand une équipe joue à domicile :FTHG = buts marqués, FTAG = buts encaissés
            buts_encaisses_dom=('FTAG', 'mean')
        )

        stats_ext = self.donnees.groupby('AwayTeam').agg(
            buts_marques_ext=('FTAG', 'mean'),    #quand une équipe joue à l'extérieur : FTAG = buts marqués, FTHG = buts encaissés
            buts_encaisses_ext=('FTHG', 'mean')
        )
        stats = stats_dom.join(stats_ext, how='outer')   #.join() fusionne les deux tableaux sur le nom d'équipe
        stats['buts_marques_moy'] = (stats['buts_marques_dom'] + stats['buts_marques_ext']) / 2           #on calcule la moyenne globale entre les matchs à domicile et extérieur
        stats['buts_encaisses_moy'] = (stats['buts_encaisses_dom'] + stats['buts_encaisses_ext']) / 2
        resultat = stats[['buts_marques_moy', 'buts_encaisses_moy']].sort_index()            #on retourne uniquement les deux colonnes finales, triées alphabétiquement
        resultat.index.name = 'equipe'    #ici on donne un nom à l'index pour améliorer la clareté

        return resultat

    def sauvegarder(self, chemin_pickle):
        """
        Sauvegarde les données nettoyées au format pickle (relecture rapide,
        conservation des types).

        :param chemin_pickle: chemin du fichier .pkl de sortie.
        :type chemin_pickle: str
        :return: rien ; écrit un fichier sur disque.
        :rtype: None
        """
        self.donnees.to_pickle(chemin_pickle)     #sauvegarde binaire Pandas
        print(f"Données sauvegardées dans : {chemin_pickle}")
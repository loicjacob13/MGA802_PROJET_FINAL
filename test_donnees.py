# test_donnees.py est unb fichier de tests unitaires pour la classe ChargeurDonnees
# Auteurs : Fabien - Loïc - Guillaume — Projet MGA802 Groupe 2
# Ce fichier est à lancer depuis la racine du projet : python -m pytest test_donnees.py -v

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from donnees import ChargeurDonnees  #donnees.py est dans le même dossier que ce fichier de test, on l'importe directement

CONTENU_CSV = (       #on crée un petit CSV factice qui va utiliser les tests (3 équipes, 6 matchs)
    "Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H\n"  # en-tête avec une colonne de cote en extra
    "01/08/2023,Arsenal,Chelsea,2,1,1.8\n"
    "05/08/2023,Chelsea,ManCity,0,3,3.1\n"
    "12/08/2023,ManCity,Arsenal,1,1,1.5\n"
    "19/08/2023,Arsenal,ManCity,2,0,2.2\n"
    "26/08/2023,Chelsea,Arsenal,1,2,2.8\n" #il n'est pas nécessaire d'avoir les csv complets pour tester les fichiers
    "02/09/2023,ManCity,Chelsea,4,1,1.4\n"  #cela augmenterai juste la durée des tests
)

def creer_csv_temporaire():
    """Écrit CONTENU_CSV dans un fichier temporaire et retourne son chemin"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                     delete=False, encoding='utf-8') as f:
        f.write(CONTENU_CSV)
        return f.name  #on retourne le chemin du fichier créé


def creer_chargeur():
    """Retourne un ChargeurDonnees déjà chargé et nettoyé, prêt à être tester"""
    chemin = creer_csv_temporaire()
    c = ChargeurDonnees(chemin)
    c.nettoyer()
    os.unlink(chemin)  #on supprime le fichier temporaire car les données sont déjà en mémoire
    return c


# Tests vérifiant le chargement des fichiers :

def test_chargement_fichier_existant():
    chemin = creer_csv_temporaire()
    c = ChargeurDonnees(chemin)
    assert isinstance(c.donnees, pd.DataFrame)  #self.donnees doit être un DataFrame
    assert len(c.donnees) > 0  #le DataFrame ne doit pas être vide
    os.unlink(chemin)  #nettoyage du fichier temporaire


def test_chargement_fichier_inexistant():
    with pytest.raises(FileNotFoundError):  #on s'attend à une erreur claire
        ChargeurDonnees("ce_fichier_nexiste_pas.csv")

def test_chargement_liste_de_fichiers():
    chemin = creer_csv_temporaire()
    c = ChargeurDonnees([chemin, chemin])  #même fichier deux fois
    assert len(c.donnees) == 12  #6 matchs × 2 = 12 lignes attendues
    os.unlink(chemin)

def test_type_invalide():
    with pytest.raises(TypeError):  #un entier n'est pas un chemin valide
        ChargeurDonnees(42)

# Tests qui vérifient le nettoyage des données :

def test_nettoyer_colonnes():
    chemin = creer_csv_temporaire()
    c = ChargeurDonnees(chemin)
    c.nettoyer()
    attendues = {'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'}
    assert set(c.donnees.columns) == attendues  #seules les 5 colonnes utiles doivent rester
    os.unlink(chemin)

def test_nettoyer_dates():
    chemin = creer_csv_temporaire()
    c = ChargeurDonnees(chemin)
    c.nettoyer()
    assert pd.api.types.is_datetime64_any_dtype(c.donnees['Date'])  #pas du texte, un vrai objet date
    os.unlink(chemin)

def test_nettoyer_buts_entiers():
    chemin = creer_csv_temporaire()
    c = ChargeurDonnees(chemin)
    c.nettoyer()
    assert c.donnees['FTHG'].dtype == int  # buts domicile en entier
    assert c.donnees['FTAG'].dtype == int  # buts extérieur en entier
    os.unlink(chemin)

# Tests sur les équipes

def test_get_equipes_nombre():
    c = creer_chargeur()
    assert len(c.get_equipes()) == 3  #notre CSV de test a exactement 3 équipes

def test_get_equipes_tri():
    c = creer_chargeur()
    equipes = c.get_equipes()
    assert equipes == sorted(equipes)  #la liste doit être triée alphabétiquement

def test_get_index_equipes_type():
    c = creer_chargeur()
    assert isinstance(c.get_index_equipes(), dict)  #doit retourner un dictionnaire

def test_get_index_equipes_valeurs():
    c = creer_chargeur()
    index = c.get_index_equipes()
    valeurs = sorted(index.values())
    assert valeurs == list(range(len(index)))  #doit être [0, 1, 2] pour 3 équipes, sans trou


# Tests de la fonction get_matchs

def test_get_matchs_type():
    c = creer_chargeur()
    assert isinstance(c.get_matchs(), np.ndarray)  #doit retourner un tableau NumPy


def test_get_matchs_forme():
    c = creer_chargeur()
    matchs = c.get_matchs()
    assert matchs.ndim == 2  #tableau à 2 dimensions
    assert matchs.shape == (6, 4)  #6 matchs, 4 valeurs par match


def test_get_matchs_dtype():
    c = creer_chargeur()
    assert np.issubdtype(c.get_matchs().dtype, np.integer)  #toutes les valeurs sont des entiers


def test_get_matchs_index_valides():
    c = creer_chargeur()
    matchs = c.get_matchs()
    n = len(c.get_equipes())
    assert matchs[:, 0].min() >= 0  #index domicile minimum >= 0
    assert matchs[:, 0].max() <= n - 1       #index domicile maximum <= n-1
    assert matchs[:, 1].min() >= 0    #index extérieur minimum >= 0
    assert matchs[:, 1].max() <= n - 1   #index extérieur maximum <= n-1


def test_get_matchs_buts_positifs():
    c = creer_chargeur()
    matchs = c.get_matchs()
    assert matchs[:, 2].min() >= 0  #buts domicile peuvent pas être négatifs
    assert matchs[:, 3].min() >= 0    #buts extérieur peuvent pas être négatifs


# Test des statistiques

def test_statistiques_equipes():
    c = creer_chargeur()
    stats = c.statistiques_equipes()
    assert isinstance(stats, pd.DataFrame)
    assert 'buts_marques_moy' in stats.columns   #colonne obligatoire
    assert 'buts_encaisses_moy' in stats.columns    #coLonne obligatoire
    assert (stats['buts_marques_moy'] >= 0).all()  #les moyennes ne peuvent pas être négatives
    assert (stats['buts_encaisses_moy'] >= 0).all()


# Tests de sauvegarder

def test_sauvegarder():
    c = creer_chargeur()
    chemin_pickle = tempfile.mktemp(suffix='.pkl')  #chemin temporaire pour le fichier pickle
    try:
        c.sauvegarder(chemin_pickle)
        assert os.path.exists(chemin_pickle)   #le fichier doit exister sur le disque
        df_relu = pd.read_pickle(chemin_pickle)
        assert len(df_relu) == len(c.donnees)     #même nombre de lignes après relecture
    finally:
        os.unlink(chemin_pickle)    #nettoyage dans tous les cas, même si le test échoue
"""
modele.py — Modèle de Poisson pour estimer les forces des équipes.
Auteurs : Fabien - Loïc - Guillaume — Projet MGA802 Groupe 2

Le modèle apprend une force d'attaque et de défense par équipe, plus un avantage
à domicile, en minimisant une perte de type Poisson sur les matchs passés.
"""
import numpy as np
import pandas as pd


class ModelePoisson:
    """Modèle de Poisson qui apprend les forces (attaque/défense) des équipes."""

    def __init__(self, n_equipes):
        """
        Initialise les paramètres du modèle dans un seul vecteur plat (pour scipy).

        :param n_equipes: nombre d'équipes différentes.
        :type n_equipes: int
        """
        self.n_equipes = n_equipes
        self.params = np.zeros(3 * n_equipes)
        # params = [attaque_0..N, defense_0..N, avantage_0..N] — vecteur plat pour scipy
        self.noms_equipes = None
        self.index_equipes = None
        self.matchs = None

    def charger_depuis_grille(self, grille):
        """
        Lit la grille MultiIndex et en extrait les matchs exploitables.

        :param grille: grille MultiIndex (format pandas) contenant les matchs.
        :type grille: pandas.DataFrame
        :return: tableau des matchs au format [idx_dom, idx_ext, buts_dom, buts_ext].
        :rtype: numpy.ndarray
        """
        matchs = grille.stack(level="Domicile", future_stack=True) #ligne de code non trouvée en cours, mais qui va permettre d'étirer un tableau, pour le mettre sous format d'une ligne
        #future_stack=True permet d'utiliser version moderne de panda
        #level="domicile" permet de basculer la ligne "domicile équipe 1"
        matchs = matchs.dropna(subset=["journee"])  # enleve la diagonale vide, après le dépliage
        matchs=matchs.reset_index() #après le strack, les collnes extérieur et domicile sont des étiquettes de ligne mais pas des vrais colonnes
        matchs=matchs.sort_values("journee").reset_index(drop=True) #cette ligne de code fait 2 choses en même temps, elle trie les matchs par numéro de journée, et le reset index est pour numéroter les lignes par 0, 1, 2

    # correspondance nom - index
        self.noms_equipes=list(grille.index) #récupère la liste des noms d'équipes
        index_equipes={nom:i for i,nom in enumerate(self.noms_equipes)} #création du dictionnaire {nom: numéro}

        #on remplace chaque nom d'équipe par son numéro puis on convertit sous format numpy
        idx_dom=matchs["Domicile"].map(index_equipes).to_numpy()
        idx_ext = matchs["Exterieur"].map(index_equipes).to_numpy()
        buts_dom = matchs["buts_dom"].to_numpy()
        buts_ext = matchs["buts_ext"].to_numpy()

# Chaque ligne = [idx_dom, idx_ext, buts_dom, buts_ext]        self.matchs = np.column_stack([idx_dom, idx_ext, buts_dom, buts_ext]).astype(int)
        return self.matchs #tableau au coeurr du problème pour entrainer le modèle


    def deplier(self, params):
        """
        Découpe le vecteur de paramètres en attaques, défenses et avantages.

        :param params: vecteur plat de longueur 3 * n_equipes.
        :type params: numpy.ndarray
        :return: tuple (attaques, defenses, avantage), chacun de longueur n_equipes.
        :rtype: tuple
        """
        n=self.n_equipes #nombre d'équipe
        attaques=params[:n]
        defenses=params[n:2*n]
        avantage=params[2*n:3*n]
        return attaques, defenses, avantage

    def calculer_lambda_mu(self, idx_dom, idx_ext):
        """
        Renvoie le nombre de buts attendus pour les deux équipes d'un match.

        :param idx_dom: index de l'équipe à domicile.
        :type idx_dom: int
        :param idx_ext: index de l'équipe à l'extérieur.
        :type idx_ext: int
        :return: tuple (lam, mu) des buts attendus à domicile et à l'extérieur.
        :rtype: tuple
        """
        # On decoupe les parametres actuels en attaques, defenses, avantage.
        attaques, defenses, avantage = self.deplier(self.params)

        # formule du modele pour l'equipe a domicile (avec le bonus d'avantage en fonction de l'équipe).
        lam = np.exp(attaques[idx_dom] - defenses[idx_ext] + avantage[idx_dom])

        # formule pour l'equipe a l'exterieur (pas de bonus domicile pour elle).
        mu = np.exp(attaques[idx_ext] - defenses[idx_dom])

        return lam, mu

    #calculer la perte pour que le modèle s'optimise

    def perte(self, params, matchs, poids=None):
        """
        Calcule la perte de Poisson (avec régularisation) sur tous les matchs.

        :param params: vecteur de paramètres à évaluer.
        :type params: numpy.ndarray
        :param matchs: tableau des matchs [idx_dom, idx_ext, buts_dom, buts_ext].
        :type matchs: numpy.ndarray
        :param poids: poids par match (même longueur que matchs). Si None, poids égaux.
        :type poids: numpy.ndarray | None
        :return: valeur de la perte totale à minimiser.
        :rtype: float
        """
        attaques, defenses, avantage = self.deplier(params)

        idx_dom = matchs[:, 0]
        idx_ext = matchs[:, 1]
        buts_dom = matchs[:, 2]
        buts_ext = matchs[:, 3]

        lam = np.exp(attaques[idx_dom] - defenses[idx_ext] + avantage[idx_dom])
        mu = np.exp(attaques[idx_ext] - defenses[idx_dom])

        erreur_dom = lam - buts_dom * np.log(lam)
        erreur_ext = mu - buts_ext * np.log(mu)

        if poids is not None:
            perte_dom = np.sum(poids * erreur_dom)
            perte_ext = np.sum(poids * erreur_ext)
        else:
            perte_dom = np.sum(erreur_dom)
            perte_ext = np.sum(erreur_ext)

        alpha = 0.001
        regularisation = alpha * (np.sum(attaques ** 2) + np.sum(defenses ** 2))

        return perte_dom + perte_ext + regularisation

#Entrainement du modèle
    def entrainer(self, matchs, poids=None):
        """
        Apprend les paramètres en minimisant la perte pondérée.

        :param matchs: tableau des matchs [idx_dom, idx_ext, buts_dom, buts_ext].
        :type matchs: numpy.ndarray
        :param poids: poids par match (même longueur que matchs). Si None, poids égaux.
        :type poids: numpy.ndarray | None
        :return: rien ; met à jour self.params et self.perte_finale.
        :rtype: None
        """
        from scipy.optimize import minimize

        depart = np.zeros(3 * self.n_equipes)

        resultat = minimize(
            lambda p: self.perte(p, matchs, poids),
            x0=depart,
            method="BFGS",
        )

        self.params = resultat.x
        self.perte_finale = resultat.fun
        return None

#sorties du code pour que ça soit utilisable pour ensuite simuler
    def get_forces(self):
        """
        Renvoie les forces apprises sous forme de dictionnaire pour le simulateur.

        :return: dictionnaire {numero_equipe: (attaque, defense)}.
        :rtype: dict
        """
        attaques,defenses,_=self.deplier(self.params)
        #il faut avant tout constrsuire le dictionnaire, et il faut aussi convertir les nombres Numpy en float
        return {i:(float(attaques[i]),float(defenses[i])) for i in range(self.n_equipes)}

    def get_avantage_domicile(self):
        """
        Renvoie l'avantage du terrain pour chaque équipe (dernier paramètre du vecteur).

        :return: dictionnaire {numero_equipe: avantage_domicile}.
        :rtype: dict
        """
        _, _, avantage = self.deplier(self.params) #les 2 premiers termes ne nous interesent pas d'où le _
        return {i: float(avantage[i]) for i in range(self.n_equipes)}

    def get_index_equipes(self):
        """
        Renvoie la correspondance {nom_equipe: numero} mémorisée à la lecture.

        :return: dictionnaire {nom_equipe: numero}, ou None si non chargé.
        :rtype: dict | None
        """
        return self.index_equipes

    def sauvegarder(self, chemin_pickle):
        """
        Sauvegarde les forces apprises au format pickle.

        :param chemin_pickle: chemin du fichier .pkl de sortie.
        :type chemin_pickle: str
        :return: rien ; écrit un fichier sur disque.
        :rtype: None
        """
        donnees = {"params": self.params, "n_equipes": self.n_equipes,
                   "noms_equipes": self.noms_equipes, "index_equipes": self.index_equipes}
        pd.to_pickle(donnees, chemin_pickle)
import numpy as np
import pandas as pd


from grille_test import construire_grille
grille = construire_grille() #on a donc la grille fictive ici

#il faut commencer à créer la classe


class ModelePoisson:
    def __init__(self,n_equipes):
        '''
        initialise les parametres du modèle,
        nécessité de créer un vecteur pour que ça soit plus simple pour scipy d'interpréter et de travailler
        :param n_equipes: nombre d'équipe différente, ici fixé à 4
        '''
        self.n_equipes = n_equipes
        self.params=np.zeros(3 * n_equipes)
        #le vecteur params consiste à appliquer une donnée force d'attaque et une donnée force défense et 1 avnatage domicile qui sera commun à tous
        #ici, il y a 3 équipes didérentes donc le vecteur params prendra juste 7 termes car le facteur de jouer à domicile est jugé uniforme à tous
        self.noms_equipes = None
        self.index_equipes = None
        self.matchs = None

# mtn il faut créer une fonction qui va lire la grille de Fabien et savoir l'utiliser pour pouvoir créer un autre vecteur params

    def charger_depuis_grille(self, grille):
        """lit la grille qui sera sous format panda et qui va extraire chaques matchs qui seront exploitables
        renvoie un tableau numpy, au lieu d'utiliser seuelement une grille multi"""
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

#mtn il faut empiler ces 4 colonnes cote à cote
# Chaque ligne = [idx_dom, idx_ext, buts_dom, buts_ext]
        self.matchs = np.column_stack([idx_dom, idx_ext, buts_dom, buts_ext]).astype(int)
        return self.matchs #tableau au coeurr du problème pour entrainer le modèle


    def deplier(self,params):
        """
        permet de découper le vecteur matchs selon en trois parties
        :param params:
        :return:
        """
        n=self.n_equipes #nombre d'équipe
        attaques=params[:n]
        defenses=params[n:2*n]
        avantage=params[2*n:3*n]
        return attaques, defenses, avantage

    #fonction qui calcule les buts attendus

    def calculer_lambda_mu(self,idx_dom,idx_ext):
        """
        renvoie le nombre de buts attendus pour les 2 équipes pour un match
        :param idx_dom:
        :param idx_ext:
        :return:
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

    '''def perte(self, params, matchs):
        attaques, defenses, avantage = self.deplier(params)

        idx_dom = matchs[:, 0]
        idx_ext = matchs[:, 1]
        buts_dom = matchs[:, 2]
        buts_ext = matchs[:, 3]

        lam = np.exp(attaques[idx_dom] - defenses[idx_ext] + avantage)
        mu = np.exp(attaques[idx_ext] - defenses[idx_dom])

        perte_dom = np.sum(lam - buts_dom * np.log(lam))
        perte_ext = np.sum(mu - buts_ext * np.log(mu))

        # ← NOUVEAU : régularisation L2 — pénalise les forces trop extrêmes
        # Sans ça, le modèle peut attribuer des forces irréalistes aux grandes équipes
        alpha = 0.001  # petit coefficient pour ne pas trop contraindre
        regularisation = alpha * (np.sum(attaques ** 2) + np.sum(defenses ** 2))

        return perte_dom + perte_ext + regularisation'''

    '''def perte(self, params, matchs):
        """
        erreur du modèle de Poisson sur tous les matchs
        :param params:
        :param matchs:
        :return:
        """
        # on découpe le vecteur params
        attaques, defenses, avantage = self._deplier(params)

        # On extrait chaque colonne du tableau de matchs.
        idx_dom = matchs[:, 0]  # colonne 0 : numeros equipes domicile
        idx_ext = matchs[:, 1]  # colonne 1 : numeros equipes exterieur
        buts_dom = matchs[:, 2]  # colonne 2 : buts reels domicile
        buts_ext = matchs[:, 3]  # colonne 3 : buts reels exterieur

        # lambda pour tous les matchs d'un coup, sans boucles for
        lam = np.exp(attaques[idx_dom] - defenses[idx_ext] + avantage)

        # mu pour tous les matchs d'un coup.
        mu = np.exp(attaques[idx_ext] - defenses[idx_dom])

        # erreur de Poisson cote domicile, différence entre la prédiction des buts avec le nombre de but réel de ce match
        # on se sert de la loi de la loi de probabilité qui donne la probabilité qu'on mette tant de buts
        # P(k buts)=(lambda^k*exp(-lambda))/(k!)
        # maximiliser la porbabilité que le modèle accorde aux vrais scores revient à maximiser la somme des logarithmes

        # avec scipy.optimize.minimize, on ne peut que minimiser une perte donc on doit minimiser son opoosé
        # le terme log(k!) n'est pas pris en compte ici car il ne dépedn que du nombre de but réel et n'est pas un indice de performance du modèle

        perte_dom = np.sum(lam - buts_dom * np.log(lam))

        # erreur de Poisson cote exterieur.
        perte_ext = np.sum(mu - buts_ext * np.log(mu))

        # Perte totale : plus elle est petite, mieux les forces collent aux vrais scores.
        return perte_dom + perte_ext'''

#Entrainement du modèle
    def entrainer(self, matchs, poids=None):
        """
        Apprend les paramètres en minimisant la perte pondérée.
        poids : tableau NumPy de même longueur que matchs (optionnel).
                Si None, tous les matchs ont le même poids.
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

    '''def entrainer(self,matchs):
        """
        apprend les parametres avec scipy et essaie d'ajuster ses valeurs pour être fidèle à chaques équipes

        :param matchs:
        :return:
        """
        from scipy.optimize import minimize #import de l'outil d'opti
        #point de départ, on fixe chaque valeurs égales à 0,
        depart=np.zeros(2 * self.n_equipes + 1)

        resultat = minimize(
            lambda p: self.perte(p, matchs),  # fonction a minimiser
            x0=depart,  # point de depart
            method="BFGS",  # methode d'optimisation classique
        )

        # On garde les paramètres appris aka le meilleur reglage trouvé.
        self.params = resultat.x

        # On garde aussi la valeur finale de la perte (pour verifier que ca a baisse).
        self.perte_finale = resultat.fun

        # entrainer ne renvoie rien : il met juste a jour self.params.
        return None'''

    '''def entrainer(self,matchs):
        """
        apprend les parametres avec scipy et essaie d'ajuster ses valeurs pour être fidèle à chaques équipes

        :param matchs:
        :return:
        """
        from scipy.optimize import minimize #import de l'outil d'opti
        #point de départ, on fixe chaque valeurs égales à 0,
        depart=np.zeros(2 * self.n_equipes + 1)

        resultat = minimize(
            lambda p: self.perte(p, matchs),  # fonction a minimiser
            x0=depart,  # point de depart
            method="BFGS",  # methode d'optimisation classique
        )

        # On garde les paramètres appris aka le meilleur reglage trouvé.
        self.params = resultat.x

        # On garde aussi la valeur finale de la perte (pour verifier que ca a baisse).
        self.perte_finale = resultat.fun

        # entrainer ne renvoie rien : il met juste a jour self.params.
        return None'''

#sorties du code pour que ça soit utilisable pour ensuite simuler
    def get_forces(self):
        """
        renvoie de façon correcte un dictinnaire de la forme suivante :
        {numero_equipe : (attaque, defense)} pour le simulateur
        :return:
        """
        attaques,defenses,_=self.deplier(self.params)
        #il faut avant tout constrsuire le dictionnaire, et il faut aussi convertir les nombres Numpy en float
        return {i:(float(attaques[i]),float(defenses[i])) for i in range(self.n_equipes)}

    def get_avantage_domicile(self):
        """
        renvoie la valeur de l'avnatage du terrain qui sera constant pour chaque équipes, il faut donc  récupérer le dernier paramètre

        :return:
        """
        _, _, avantage = self.deplier(self.params) #les 2 premiers termes ne nous interesent pas d'où le _
        return {i: float(avantage[i]) for i in range(self.n_equipes)}
    def get_index_equipes(self):
        return self.index_equipes

    def sauvegarder(self, chemin_pickle):
        """Sauvegarde les forces apprises au format pickle."""
        donnees = {"params": self.params, "n_equipes": self.n_equipes,
                   "noms_equipes": self.noms_equipes, "index_equipes": self.index_equipes}
        pd.to_pickle(donnees, chemin_pickle)

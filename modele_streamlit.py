import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# CORRECTION : Import commenté pour éviter un crash dans Streamlit
# from grille_test import construire_grille
# grille = construire_grille() # on a donc la grille fictive ici
# ----------------------------------------------------------------------

class ModelePoisson:
    def __init__(self, n_equipes):
        '''
        initialise les parametres du modèle,
        nécessité de créer un vecteur pour que ça soit plus simple pour scipy d'interpréter et de travailler
        :param n_equipes: nombre d'équipe différente, ici fixé à 4
        '''
        self.n_equipes = n_equipes
        self.params = np.zeros(2*n_equipes+1)
        self.noms_equipes = None
        self.index_equipes = None
        self.matchs = None

    def charger_depuis_grille(self, grille):
        """lit la grille qui sera sous format panda et qui va extraire chaques matchs qui seront exploitables"""
        matchs = grille.stack(level="Domicile", future_stack=True)
        matchs = matchs.dropna(subset=["journee"])
        matchs = matchs.reset_index()
        matchs = matchs.sort_values("journee").reset_index(drop=True)

        self.noms_equipes = list(grille.index)
        index_equipes = {nom: i for i, nom in enumerate(self.noms_equipes)}

        idx_dom = matchs["Domicile"].map(index_equipes).to_numpy()
        idx_ext = matchs["Exterieur"].map(index_equipes).to_numpy()
        buts_dom = matchs["buts_dom"].to_numpy()
        buts_ext = matchs["buts_ext"].to_numpy()

        self.matchs = np.column_stack([idx_dom, idx_ext, buts_dom, buts_ext]).astype(int)
        return self.matchs

    def deplier(self, params):
        """permet de découper le vecteur matchs selon en trois parties"""
        n = self.n_equipes
        attaques = params[:n]
        defenses = params[n:2*n]
        avantage = params[2*n]
        return attaques, defenses, avantage

    def calculer_lambda_mu(self, idx_dom, idx_ext):
        """renvoie le nombre de buts attendus pour les 2 équipes pour un match"""
        attaques, defenses, avantage = self.deplier(self.params)
        lam = np.exp(attaques[idx_dom] - defenses[idx_ext] + avantage)
        mu = np.exp(attaques[idx_ext] - defenses[idx_dom])
        return lam, mu

    def perte(self, params, matchs, poids=None):
        attaques, defenses, avantage = self.deplier(params)

        idx_dom = matchs[:, 0]
        idx_ext = matchs[:, 1]
        buts_dom = matchs[:, 2]
        buts_ext = matchs[:, 3]

        lam = np.exp(attaques[idx_dom] - defenses[idx_ext] + avantage)
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

    def entrainer(self, matchs, poids=None):
        """Apprend les paramètres en minimisant la perte pondérée."""
        from scipy.optimize import minimize

        depart = np.zeros(2 * self.n_equipes + 1)

        resultat = minimize(
            lambda p: self.perte(p, matchs, poids),
            x0=depart,
            method="BFGS",
        )

        self.params = resultat.x
        self.perte_finale = resultat.fun
        return None

    def get_forces(self):
        """renvoie un dictionnaire {numero_equipe : (attaque, defense)}"""
        attaques, defenses, _ = self.deplier(self.params)
        return {i: (float(attaques[i]), float(defenses[i])) for i in range(self.n_equipes)}

    def get_avantage_domicile(self):
        """renvoie la valeur de l'avantage du terrain"""
        _, _, avantage = self.deplier(self.params)
        return float(avantage)

    def get_index_equipes(self):
        return self.index_equipes

    def sauvegarder(self, chemin_pickle):
        """Sauvegarde les forces apprises au format pickle."""
        donnees = {"params": self.params, "n_equipes": self.n_equipes,
                   "noms_equipes": self.noms_equipes, "index_equipes": self.index_equipes}
        pd.to_pickle(donnees, chemin_pickle)
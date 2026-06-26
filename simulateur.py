"""
simulateur.py — Simulation Monte-Carlo de la Premier League par loi de Poisson.
Auteurs : Fabien - Loïc - Guillaume — Projet MGA802 Groupe 2
"""
import numpy as np
import pandas as pd

class Simulateur:
    """Simule des saisons de Premier League par tirage aléatoire (loi de Poisson)."""
    def __init__(self, forces: dict, avantage_domicile: float, index_equipes: dict):
        self.forces = forces                        # {index: (attaque, defense)}
        self.avantage_domicile = avantage_domicile  # float
        self.index_equipes = index_equipes          # {nom: index}
        self.n_equipes = len(forces)

        # Dictionnaire inverse {index: nom} pour afficher les vrais noms dans les résultats
        self.noms_equipes = {v: k for k, v in index_equipes.items()}

        # Générateur aléatoire NumPy (reproductible si on fixe la graine)
        self.rng = np.random.default_rng()

        # Tableaux NumPy des forces pour accès vectorisé rapide
        # attaques[i] = force d'attaque de l'équipe i
        self.attaques = np.array([forces[i][0] for i in range(self.n_equipes)])
        self.defenses = np.array([forces[i][1] for i in range(self.n_equipes)])
        self.avantages = np.array([avantage_domicile[i] for i in range(self.n_equipes)])

    # ------------------------------------------------------------------
    # ÉTAPE 1 — Simuler un seul match
    # ------------------------------------------------------------------

    def simuler_match(self, index_dom: int, index_ext: int) -> tuple:
        """Tire un score aléatoire pour un match entre deux équipes."""
        attaque_dom = self.attaques[index_dom]
        defense_dom = self.defenses[index_dom]
        attaque_ext = self.attaques[index_ext]
        defense_ext = self.defenses[index_ext]

        # Formule du modèle de Poisson (voir brief_guillaume.md §2)
        lam = np.exp(attaque_dom - defense_ext + self.avantages[index_dom])  # buts attendus dom.
        mu  = np.exp(attaque_ext - defense_dom)                           # buts attendus ext.

        # Tirage aléatoire selon la loi de Poisson
        buts_dom = int(self.rng.poisson(lam))
        buts_ext = int(self.rng.poisson(mu))

        return buts_dom, buts_ext

    # ------------------------------------------------------------------
    # ÉTAPE 2 — Simuler une saison complète (380 matchs)
    # ------------------------------------------------------------------

    def simuler_saison(self) -> pd.DataFrame:
        """Joue les 380 matchs d'une saison et retourne le classement final."""
        # --- Initialisation des compteurs (tableaux NumPy, plus rapide que des listes) ---
        points    = np.zeros(self.n_equipes, dtype=int)   # points cumulés
        diff_buts = np.zeros(self.n_equipes, dtype=int)   # différence de buts cumulée

        # --- Génération du calendrier : toutes les paires aller-retour ---
        # On génère les 380 matchs (19 journées aller + 19 journées retour)
        # et on les mélange pour simuler un ordre de saison aléatoire.
        calendrier = []
        for dom in range(self.n_equipes):
            for ext in range(self.n_equipes):
                if dom != ext:
                    calendrier.append((dom, ext))

        # Mélange aléatoire de l'ordre des matchs (reproduit la variabilité d'une vraie saison)
        calendrier = np.array(calendrier)                        # shape (380, 2)
        ordre = self.rng.permutation(len(calendrier))            # indices mélangés
        calendrier = calendrier[ordre]                           # on réordonne

        # Jeu de chaque match et attribution des points
        for dom, ext in calendrier:
            buts_dom, buts_ext = self.simuler_match(int(dom), int(ext))

            diff = buts_dom - buts_ext
            diff_buts[dom] += diff    # bonne différence pour le domicile
            diff_buts[ext] -= diff    # inverse pour l'extérieur

            if buts_dom > buts_ext:       # victoire à domicile
                points[dom] += 3
            elif buts_dom == buts_ext:    # match nul
                points[dom] += 1
                points[ext] += 1
            else:                         # victoire à extérieur
                points[ext] += 3

        # Construction du classement dans un DataFrame
        classement = pd.DataFrame({
            'equipe'    : [self.noms_equipes[i] for i in range(self.n_equipes)],
            'points'    : points,
            'diff_buts' : diff_buts,
        })

        # Tri par points décroissants, puis différence de buts en cas d'égalité
        classement = classement.sort_values(
            by=['points', 'diff_buts'],
            ascending=[False, False]
        ).reset_index(drop=True)

        # La position dans le classement va de 1 (champion) à 20 (dernier)
        classement['position'] = classement.index + 1

        return classement

    # ------------------------------------------------------------------
    # ÉTAPE 3 — Simulation Monte-Carlo (N saisons → probabilités)
    # ------------------------------------------------------------------
    def simuler_monte_carlo(self, n_simulations: int) -> pd.DataFrame:
        positions_acc = np.zeros((self.n_equipes, n_simulations), dtype=int)
        points_acc = np.zeros((self.n_equipes, n_simulations), dtype=int)

        for sim in range(n_simulations):
            classement = self.simuler_saison()
            for _, ligne in classement.iterrows():
                idx = self.index_equipes[ligne['equipe']]
                positions_acc[idx, sim] = ligne['position']
                points_acc[idx, sim] = ligne['points']

        #points bruts accessibles pour graphique_distribution_points
        self.points_bruts = {
            self.noms_equipes[i]: points_acc[i] for i in range(self.n_equipes)
        }

        proba_titre = (positions_acc == 1).mean(axis=1)
        proba_top4 = (positions_acc <= 4).mean(axis=1)
        proba_relegation = (positions_acc >= 18).mean(axis=1)
        position_moyenne = positions_acc.mean(axis=1)
        points_moyens = points_acc.mean(axis=1)

        resultats = pd.DataFrame({
            'proba_titre': proba_titre,
            'proba_top4': proba_top4,
            'proba_relegation': proba_relegation,
            'position_moyenne': position_moyenne,
            'points_moyens': points_moyens,
        }, index=[self.noms_equipes[i] for i in range(self.n_equipes)])

        resultats.index.name = 'equipe'
        resultats = resultats.sort_values('position_moyenne')

        return resultats

    '''def simuler_monte_carlo(self, n_simulations: int) -> pd.DataFrame:
        """Répète la simulation de saison N fois et calcule les probabilités de classement."""
        # Tableaux NumPy pour accumuler les résultats — évite les listes Python lentes
        # positions_acc[i, sim] = position finale de l'équipe i à la simulation sim
        positions_acc = np.zeros((self.n_equipes, n_simulations), dtype=int)
        points_acc    = np.zeros((self.n_equipes, n_simulations), dtype=int)

        # Dictionnaire nom → index pour retrouver la ligne dans les tableaux
        nom_vers_index = self.index_equipes   # {nom: index}

        for sim in range(n_simulations):
            classement = self.simuler_saison()  # DataFrame trié par points décroissants

            # On enregistre la position et les points de chaque équipe
            for _, ligne in classement.iterrows():
                idx = nom_vers_index[ligne['equipe']]
                positions_acc[idx, sim] = ligne['position']
                points_acc[idx, sim]    = ligne['points']

        # --- Calcul vectorisé des probabilités avec NumPy ---
        # proba_titre : proportion de simulations où l'équipe finit 1ère
        proba_titre      = (positions_acc == 1).mean(axis=1)
        # proba_top4 : proportion de simulations où l'équipe finit dans les 4 premières
        proba_top4       = (positions_acc <= 4).mean(axis=1)
        # proba_relegation : proportion de simulations où l'équipe finit 18e, 19e ou 20e
        proba_relegation = (positions_acc >= 18).mean(axis=1)
        # position_moyenne : moyenne des positions finales
        position_moyenne = positions_acc.mean(axis=1)
        # points_moyens : moyenne des points finaux
        points_moyens    = points_acc.mean(axis=1)

        # --- Construction du DataFrame résultat ---
        resultats = pd.DataFrame({
            'proba_titre'     : proba_titre,
            'proba_top4'      : proba_top4,
            'proba_relegation': proba_relegation,
            'position_moyenne': position_moyenne,
            'points_moyens'   : points_moyens,
        }, index=[self.noms_equipes[i] for i in range(self.n_equipes)])

        resultats.index.name = 'equipe'

        # Tri par position moyenne croissante pour faciliter la lecture
        resultats = resultats.sort_values('position_moyenne')

        return resultats'''

    # ------------------------------------------------------------------
    # ÉTAPE 4 — Facteur de moral
    # ------------------------------------------------------------------

    def appliquer_moral(self, lambda_base: float, index_equipe: int,
                        journee: int, classement_actuel: pd.DataFrame) -> float:
        """Ajuste le lambda d'une équipe en fin de saison selon son enjeu."""
        # Avant la journée 35, aucun ajustement
        if journee < 35:
            return lambda_base

        multiplicateur = 1.0  # par défaut : aucun bonus

        # On cherche la position actuelle de cette équipe dans le classement
        nom = self.noms_equipes[index_equipe]

        # On cherche la ligne de cette équipe dans le classement actuel
        ligne = classement_actuel[classement_actuel['equipe'] == nom]

        if ligne.empty:
            return lambda_base   # sécurité : si l'équipe n'est pas trouvée

        position = int(ligne['position'].values[0])

        # Équipe menacée de relégation (18e, 19e ou 20e) → se bat → bonus
        if position >= 18:
            multiplicateur = 1.1   # +10% de motivation

        # Équipe en course pour le titre ou le top 4 → motivation européenne
        elif position <= 4:
            multiplicateur = 1.05  # +5% de motivation

        return lambda_base * multiplicateur
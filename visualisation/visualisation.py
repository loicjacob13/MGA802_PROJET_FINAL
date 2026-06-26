"""
visualisation.py — Graphiques Matplotlib pour les résultats de simulation.
Auteurs : Fabien - Loïc - Guillaume — Projet MGA802 Groupe 2
"""
import matplotlib.pyplot as plt
import pandas as pd

class Visualiseur:
    """Affiche les résultats de la simulation Monte-Carlo avec Matplotlib."""

    def __init__(self, resultats: pd.DataFrame):
        """
        Initialise le visualiseur avec les résultats de simulation.

        :param resultats: classement Monte-Carlo indexé par nom d'équipe (sortie de
            simuler_monte_carlo).
        :type resultats: pandas.DataFrame
        """
        self.resultats = resultats   # DataFrame indexé par nom d'équipe

    # ------------------------------------------------------------------
    # Graphique 1 — Probabilité d'être champion
    # ------------------------------------------------------------------
    def graphique_probabilites_titre(self) -> None:
        """
        Trace un diagramme en barres de la probabilité d'être champion par équipe.

        :return: rien ; prépare la figure Matplotlib (affichée par plt.show()).
        :rtype: None
        """
        df = self.resultats.sort_values('proba_titre', ascending=False)
        noms = df.index.tolist()
        probas = df['proba_titre'] * 100

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.bar(noms, probas, color='steelblue', edgecolor='white', linewidth=0.5)

        for i, val in enumerate(probas):
            if val > 0.5:
                ax.text(i, val + 0.3, f"{val:.1f}%", ha='center', fontsize=8)

        ax.set_title("Probabilité d'être champion de Premier League",
                     fontsize=14, fontweight='bold')
        ax.set_ylabel("Probabilité (%)")
        ax.set_xlabel("Équipe")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.draw()

    # ------------------------------------------------------------------
    # Graphique 2 — Distribution des points d'une équipe
    # ------------------------------------------------------------------
    def graphique_distribution_points(self, nom_equipe: str, simulateur=None) -> None:
        """
        Trace l'histogramme des points finaux d'une équipe sur toutes les simulations.

        Si un simulateur est fourni, utilise la vraie distribution (points_bruts) ;
        sinon, génère une approximation gaussienne autour de la moyenne.

        :param nom_equipe: nom de l'équipe à détailler.
        :type nom_equipe: str
        :param simulateur: simulateur ayant l'attribut points_bruts (optionnel).
        :type simulateur: Simulateur | None
        :raises ValueError: si l'équipe est introuvable dans les résultats.
        :return: rien ; prépare la figure Matplotlib (affichée par plt.show()).
        :rtype: None
        """
        if nom_equipe not in self.resultats.index:
            raise ValueError(f"Équipe '{nom_equipe}' introuvable.")

        fig, ax = plt.subplots(figsize=(10, 5))

        if simulateur is not None and hasattr(simulateur, 'points_bruts'):
            # Vrais points simulés — distribution réelle
            points = simulateur.points_bruts[nom_equipe]
            ax.hist(points, bins=20, color='steelblue', edgecolor='white')
            moyenne = points.mean()
        else:
            # Fallback approximatif si pas de simulateur fourni
            moyenne = self.resultats.loc[nom_equipe, 'points_moyens']
            import numpy as np
            points = np.random.default_rng(0).normal(moyenne, 10, 500).clip(0, 114)
            ax.hist(points, bins=20, color='steelblue', edgecolor='white')

        ax.axvline(moyenne, color='crimson', linewidth=2, label=f"Moyenne : {moyenne:.1f} pts")
        ax.set_title(f"Distribution des points finaux — {nom_equipe}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Points en fin de saison")
        ax.set_ylabel("Nombre de simulations")
        ax.legend()
        plt.tight_layout()
        plt.draw()

    # ------------------------------------------------------------------
    # Graphique 3 — Classement moyen de toutes les équipes
    # ------------------------------------------------------------------
    def graphique_classement_moyen(self) -> None:
        """
        Trace les positions moyennes de toutes les équipes en barres horizontales.

        Un code couleur distingue le top 4 (bleu), la zone de relégation (rouge)
        et le milieu de tableau (gris).

        :return: rien ; prépare la figure Matplotlib (affichée par plt.show()).
        :rtype: None
        """
        # Tri par position_moyenne CROISSANTE puis on inverse l'ordre pour barh
        # (barh met la dernière valeur en haut → on trie décroissant pour avoir le 1er en haut)
        df = self.resultats.sort_values('position_moyenne', ascending=False)  # décroissant !

        noms = df.index.tolist()
        positions = df['position_moyenne'].tolist()

        couleurs = []
        for pos in positions:
            if pos <= 4:
                couleurs.append('steelblue')
            elif pos >= 18:
                couleurs.append('crimson')
            else:
                couleurs.append('lightgray')

        fig, ax = plt.subplots(figsize=(10, 10))
        barres = ax.barh(noms, positions, color=couleurs, edgecolor='white')

        for barre, valeur in zip(barres, positions):
            ax.text(valeur + 0.1, barre.get_y() + barre.get_height() / 2,
                    f"{valeur:.1f}", va='center', fontsize=9)

        ax.axvline(4.5, color='steelblue', linestyle='--', linewidth=1, label='Limite top 4')
        ax.axvline(17.5, color='crimson', linestyle='--', linewidth=1, label='Limite relégation')

        ax.set_title("Classement moyen des équipes (Monte-Carlo)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Position moyenne en fin de saison", fontsize=11)
        ax.set_ylabel("Équipe", fontsize=11)
        # Axe X : 1 à gauche (meilleur), 20 à droite — PAS d'inversion cette fois
        ax.set_xlim(0, 21)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.draw()
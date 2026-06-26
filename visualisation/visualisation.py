"""
visualisation.py — Graphiques Matplotlib pour les résultats de simulation.
Auteurs : Fabien - Loïc - Guillaume — Projet MGA802 Groupe 2
"""
import matplotlib.pyplot as plt
import pandas as pd

class Visualiseur:
    """Affiche les résultats de la simulation Monte-Carlo avec Matplotlib."""

    def __init__(self, resultats: pd.DataFrame):
        self.resultats = resultats   # DataFrame indexé par nom d'équipe

    # ------------------------------------------------------------------
    # Graphique 1 — Probabilité d'être champion
    # ------------------------------------------------------------------
    def graphique_probabilites_titre(self) -> None:
        """Diagramme en barres : probabilité de chaque équipe d'être championne."""
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
        #plt.savefig("../graphique_probabilites_titre.png", dpi=150)
        plt.draw()

    '''def graphique_probabilites_titre(self) -> None:
        """Diagramme en barres : probabilité de chaque équipe d'être championne."""
        # Tri par proba_titre décroissante
        df = self.resultats.sort_values('proba_titre', ascending=False)

        noms   = df.index.tolist()             # noms des équipes sur l'axe X
        probas = df['proba_titre'] * 100       # conversion en pourcentage

        fig, ax = plt.subplots(figsize=(14, 6))

        barres = ax.bar(noms, probas, color='steelblue', edgecolor='white', linewidth=0.5)

        # On affiche la valeur au-dessus de chaque barre (si > 0.5 %)
        for barre, valeur in zip(barres, probas):
            if valeur > 0.5:
                ax.text(
                    barre.get_x() + barre.get_width() / 2,  # centré sur la barre
                    barre.get_height() + 0.3,               # légèrement au-dessus
                    f"{valeur:.1f}%",
                    ha='center', va='bottom', fontsize=8
                )

        ax.set_title("Probabilité d'être champion de Premier League", fontsize=14, fontweight='bold')
        ax.set_xlabel("Équipe", fontsize=11)
        ax.set_ylabel("Probabilité (%)", fontsize=11)
        ax.set_ylim(0, max(probas) * 1.15 + 1)    # marge en haut pour les étiquettes
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.tight_layout()
        plt.savefig("graphique_probabilites_titre.png", dpi=150)
        plt.show()'''

    # ------------------------------------------------------------------
    # Graphique 2 — Distribution des points d'une équipe
    # ------------------------------------------------------------------
    def graphique_distribution_points(self, nom_equipe: str, simulateur=None) -> None:
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
        #plt.savefig(f"graphique_distribution_{nom_equipe.replace(' ', '_')}.png", dpi=150)
        plt.draw()

    '''def graphique_distribution_points(self, nom_equipe: str) -> None:
        """Histogramme des points finaux d'une équipe sur toutes les simulations."""
        # Vérification que l'équipe existe bien dans les résultats
        if nom_equipe not in self.resultats.index:
            noms_disponibles = list(self.resultats.index)
            raise ValueError(
                f"Équipe '{nom_equipe}' introuvable. "
                f"Équipes disponibles : {noms_disponibles}"
            )

        points_moyens = self.resultats.loc[nom_equipe, 'points_moyens']

        # Note : simuler_monte_carlo() ne stocke que les moyennes agrégées,
        # pas la distribution complète match par match.
        # Pour cet histogramme, on affiche donc la distribution simulée
        # via un DataFrame enrichi si disponible, sinon un message explicatif.
        # Pour une vraie distribution, il faudrait passer points_acc en attribut.

        # --- Solution pratique : re-simuler rapidement N points pour cette équipe ---
        # (cela fonctionne sans modifier l'interface de simuler_monte_carlo)
        fig, ax = plt.subplots(figsize=(10, 5))

        # On crée une distribution gaussienne approximative autour de la moyenne
        # à partir des données agrégées disponibles.
        # C'est une approximation visuelle acceptable pour l'affichage.
        import numpy as np
        rng = np.random.default_rng(0)
        # Écart-type estimé d'une saison de football : environ 8-12 points
        points_simules = rng.normal(loc=points_moyens, scale=10, size=500).clip(0, 114)

        ax.hist(points_simules, bins=20, color='steelblue', edgecolor='white',
                linewidth=0.5, density=False)

        ax.axvline(points_moyens, color='crimson', linewidth=2,
                   label=f"Moyenne : {points_moyens:.1f} pts")

        ax.set_title(f"Distribution des points finaux — {nom_equipe}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Points en fin de saison", fontsize=11)
        ax.set_ylabel("Nombre de simulations", fontsize=11)
        ax.legend(fontsize=10)
        plt.tight_layout()
        nom_fichier = f"graphique_distribution_{nom_equipe.replace(' ', '_')}.png"
        #plt.savefig(nom_fichier, dpi=150)
        plt.show()'''

    # ------------------------------------------------------------------
    # Graphique 3 — Classement moyen de toutes les équipes
    # ------------------------------------------------------------------
    def graphique_classement_moyen(self) -> None:
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
        #plt.savefig("../graphique_classement_moyen.png", dpi=150)
        plt.draw()

    '''def graphique_classement_moyen(self) -> None:
        """Barres horizontales : position moyenne de chaque équipe sur toutes les simulations.

        Les équipes sont triées par position moyenne croissante (le meilleur en haut).
        Un code couleur distingue le top 4 (bleu), la zone de relégation (rouge)
        et le reste (gris).
        """
        # Tri par position_moyenne croissante (meilleure équipe en premier)
        df = self.resultats.sort_values('position_moyenne', ascending=True)

        noms      = df.index.tolist()
        positions = df['position_moyenne'].tolist()

        # Code couleur selon la zone du classement
        couleurs = []
        for pos in positions:
            if pos <= 4:
                couleurs.append('steelblue')    # top 4 (Europe)
            elif pos >= 18:
                couleurs.append('crimson')       # relégation
            else:
                couleurs.append('lightgray')     # milieu de tableau

        fig, ax = plt.subplots(figsize=(10, 10))

        # barh : barres horizontales — l'axe Y porte les noms, X la position
        barres = ax.barh(noms, positions, color=couleurs, edgecolor='white', linewidth=0.5)

        # Étiquette numérique à droite de chaque barre
        for barre, valeur in zip(barres, positions):
            ax.text(
                valeur + 0.1,                              # légèrement à droite
                barre.get_y() + barre.get_height() / 2,   # centré verticalement
                f"{valeur:.1f}",
                va='center', fontsize=9
            )

        # Ligne de référence pour le top 4 et la zone de relégation
        ax.axvline(4.5,  color='steelblue', linestyle='--', linewidth=1,
                   label='Limite top 4 (Europe)')
        ax.axvline(17.5, color='crimson',   linestyle='--', linewidth=1,
                   label='Limite relégation')

        ax.set_title("Classement moyen des équipes (Monte-Carlo)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Position moyenne en fin de saison", fontsize=11)
        ax.set_ylabel("Équipe", fontsize=11)
        ax.invert_xaxis()   # position 1 à gauche (meilleure), 20 à droite
        ax.set_xlim(20.5, 0)
        ax.legend(fontsize=9, loc='lower left')
        plt.tight_layout()
        plt.savefig("graphique_classement_moyen.png", dpi=150)
        plt.show()'''

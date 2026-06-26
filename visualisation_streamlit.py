"""visualisation_streamlit.py — Graphiques Plotly pour l'interface Streamlit.
Auteurs : Fabien - Loïc - Guillaume — Projet MGA802 Groupe 2
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


class Visualiseur:
    def __init__(self, resultats: pd.DataFrame):
        # On trie pour avoir les meilleures équipes en haut par défaut
        self.resultats = resultats.sort_values('position_moyenne', ascending=True)

    def graphique_probabilites_titre(self):
        """Graphique à barres horizontales strict."""
        df = self.resultats[self.resultats['proba_titre'] > 0].sort_values('proba_titre', ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=df.index,
            x=df['proba_titre'],
            orientation='h',
            marker=dict(color='#000000'),
            text=df['proba_titre'].apply(lambda x: f"{x * 100:.1f}%"),
            textposition='outside',
            textfont=dict(family="Arial", size=12, color="#000000")
        ))
        fig.update_layout(
            title=dict(text="<b>Probabilités de titre de champion</b>",
                       font=dict(family="Arial", size=16, color="#000000")),
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            showlegend=False,
            height=400,
            margin=dict(l=100, r=40, t=50, b=10),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, tickfont=dict(family="Arial", size=12, color="#000000"))
        )
        return fig

    def graphique_classement_moyen(self):
        """Ligne de trajectoire conventionnelle (Gauche->Droite, Haut->Bas)."""
        df = self.resultats.sort_values('position_moyenne', ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['position_moyenne'],
            y=df.index,
            mode='markers+text',
            marker=dict(color='#000000', size=10, symbol='square'),
            text=df['position_moyenne'].round(1),
            textposition="top right",
            textfont=dict(family="Arial", size=11, color="#000000")
        ))

        # Lignes verticales de repère
        fig.add_vline(x=4.5, line_width=2, line_dash="dash", line_color="#888888")
        fig.add_vline(x=17.5, line_width=2, line_dash="dash", line_color="#888888")

        fig.update_layout(
            title=dict(text="<b>Position moyenne au classement final</b>",
                       font=dict(family="Arial", size=16, color="#000000")),
            xaxis=dict(
                title=dict(text="Position", font=dict(color="#000000")),
                range=[0.5, 20.5],  # Lecture de 1 à 20 de gauche à droite
                dtick=1,
                side="top",
                gridcolor='#e0e0e0',
                tickfont=dict(color="#000000")
            ),
            yaxis=dict(
                autorange="reversed",  # Le 1er est en haut, le 20e en bas
                showgrid=True,
                gridcolor='#f0f0f0',
                tickfont=dict(color="#000000")
            ),
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            height=600,
            margin=dict(l=100, r=40, t=80, b=20)
        )
        return fig

    def graphique_distribution_points(self, nom_equipe: str, simulateur):
        points = simulateur.points_bruts[nom_equipe]
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=points,
            histnorm='probability',
            marker=dict(color='#888888', line=dict(color='#000000', width=1)),
            xbins=dict(start=min(points) - 0.5, end=max(points) + 0.5, size=1)
        ))
        moyenne = points.mean()
        fig.add_vline(x=moyenne, line_width=3, line_color="#000000")
        fig.update_layout(
            title=dict(text=f"<b>Distribution des points: {nom_equipe}</b>", font=dict(color="#000000")),
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            yaxis=dict(showgrid=True, gridcolor='#e0e0e0', tickfont=dict(color="#000000")),
            xaxis=dict(showgrid=False, tickfont=dict(color="#000000")),
            height=350,
            margin=dict(l=50, r=20, t=50, b=40),
            font=dict(color="#000000")
        )
        return fig
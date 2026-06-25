"""
main_streamlit.py - Interface brutaliste.
Entraînement sur l'historique complet (8 saisons), intégration des promus/relégués, et simulation de la nouvelle saison.
"""
import streamlit as st
import numpy as np
import pandas as pd
from donnees import ChargeurDonnees
from modele import ModelePoisson
from simulateur import Simulateur
from visualisation_streamlit import Visualiseur
from forces_promus import forces_pour_position_cible

st.set_page_config(page_title="Simulateur Premier League", layout="wide", initial_sidebar_state="collapsed")

# Injection CSS pour le design brutaliste monochrome et correction des bugs d'affichage
st.markdown("""
<style>
    .stApp, .main, [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    p, span, h1, h2, h3, h4, h5, h6, label, div {
        color: #000000 !important;
    }
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stSidebar"] { display: none !important; }
    header { display: none !important; }
    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }

    /* Sélecteurs et champs de saisie */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background-color: #ffffff !important;
        border: 2px solid #000000 !important;
        border-radius: 0px !important;
    }

    /* CORRECTION : Forcer le texte en noir dans les champs de saisie numériques */
    div[data-baseweb="input"] input, .stNumberInput input {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-weight: bold !important;
    }

    ul[role="listbox"], li[role="option"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    /* Boutons industriels */
    div.stButton > button {
        border: 2px solid #000000 !important;
        border-radius: 0px !important;
        background-color: #ffffff !important;
        font-weight: 900 !important;
        text-transform: uppercase;
        padding: 1rem !important;
    }

    /* CORRECTION : Fond noir et texte blanc au survol */
    div.stButton > button:hover, div.stButton > button:hover * {
        background-color: #000000 !important;
        color: #ffffff !important;
        border-color: #000000 !important;
    }

    /* Métriques entourées de noir */
    div[data-testid="stMetric"] {
        border: 2px solid #000000 !important;
        background-color: #ffffff !important;
        padding: 15px !important;
    }
    div[data-testid="stMetricValue"] { font-weight: bold !important; }
    hr { border-bottom-color: #000000 !important; border-bottom-width: 2px !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def executer_calcul_modele(p1, p2, p3, p4, p5, p6, p7, p8):
    """Charge les 8 saisons, entraîne le modèle universel, et extrait les 20 équipes de 25-26."""
    fichiers = [
        "2018-2019.csv", "2019-2020.csv", "2020-2021.csv", "2021-2022.csv",
        "2022-2023.csv", "2023-2024.csv", "2024-2025.csv", "2025-2026.csv"
    ]

    # 1. Entraînement universel sur tout l'historique
    c_univ = ChargeurDonnees(fichiers)
    c_univ.nettoyer()
    index_univ = c_univ.get_index_equipes()
    matchs_univ = c_univ.get_matchs()

    # 2. Calcul du vecteur de poids via la date chronologique des matchs
    annees_debut = np.where(
        c_univ.donnees['Date'].dt.month >= 7,
        c_univ.donnees['Date'].dt.year,
        c_univ.donnees['Date'].dt.year - 1
    )

    poids_global = np.zeros(len(annees_debut))
    pcts = [p1, p2, p3, p4, p5, p6, p7, p8]
    annees = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

    for annee, pct in zip(annees, pcts):
        mask = (annees_debut == annee)
        nb = mask.sum()
        if nb > 0:
            poids_global[mask] = (pct / 100.0) / nb

    # 3. Exécution de l'optimisation
    modele = ModelePoisson(len(index_univ))
    modele.entrainer(matchs_univ, poids=poids_global)
    forces_univ = modele.get_forces()
    avantage_univ = modele.get_avantage_domicile()   # index UNIVERSEL

    # 4. Isolation stricte des 20 équipes de la saison actuelle pour la base de simulation
    c_actuel = ChargeurDonnees("2025-2026.csv")
    c_actuel.nettoyer()
    equipes_actuelles = sorted(c_actuel.get_equipes())

    index_simu = {nom: i for i, nom in enumerate(equipes_actuelles)}
    forces_simu = {}
    avantage_simu = {}   # CORRECTION : on réindexe AUSSI l'avantage sur 0-19
    for nom in equipes_actuelles:
        idx_u = index_univ[nom]      # numéro dans l'index universel
        idx_s = index_simu[nom]      # numéro dans l'index des 20 équipes
        forces_simu[idx_s] = forces_univ[idx_u]
        avantage_simu[idx_s] = avantage_univ[idx_u]   # CORRECTION : même réindexation que les forces

    return forces_simu, avantage_simu, index_simu


# --- DÉFINITION DES MOUVEMENTS (Relégués / Promus) ---
RELEGUES = ["West Ham", "Burnley", "Wolves"]
PROMUS = [("Coventry", 15.0), ("Ipswich", 16.9), ("Millwall", 17.4)]

# --- STRUCTURE ASYMÉTRIQUE ---
col_controle, col_affichage = st.columns([1, 3], gap="large")

with col_controle:
    st.markdown("## CONTRÔLE")
    st.markdown("---")

    n_simulations = st.number_input("Simulations Monte-Carlo", 100, 5000, 500, 100)

    st.markdown("### Poids de l'historique (%)")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        p_18 = st.number_input("18-19", 0, 100, 2)
        p_19 = st.number_input("19-20", 0, 100, 3)
        p_20 = st.number_input("20-21", 0, 100, 5)
        p_21 = st.number_input("21-22", 0, 100, 10)
    with col_p2:
        p_22 = st.number_input("22-23", 0, 100, 10)
        p_23 = st.number_input("23-24", 0, 100, 20)
        p_24 = st.number_input("24-25", 0, 100, 20)
        p_25 = st.number_input("25-26", 0, 100, 30)

    total_poids = p_18 + p_19 + p_20 + p_21 + p_22 + p_23 + p_24 + p_25

    if total_poids != 100:
        st.error(f"Total : {total_poids}%. Requis : 100%.")
        calcul_autorise = False
    else:
        st.success("Pondération validée.")
        calcul_autorise = True

    # Récupération silencieuse des forces de base
    forces_base, avantage_base, index_base = executer_calcul_modele(p_18, p_19, p_20, p_21, p_22, p_23, p_24, p_25)

    # Génération de la liste des équipes pour la NOUVELLE saison
    noms_base = list(index_base.keys())
    equipes_nouvelle_saison = sorted([nom for nom in noms_base if nom not in RELEGUES] + [p[0] for p in PROMUS])

    st.markdown("<br>", unsafe_allow_html=True)
    equipe_choisie = st.selectbox("Choisissez votre équipe", equipes_nouvelle_saison)

    st.markdown("<br>", unsafe_allow_html=True)

    # Affichage des relégués et promus
    st.markdown("##### INFOS SAISON")
    st.caption(f"**Relégués :** {', '.join(RELEGUES)}")
    st.caption(f"**Promus :** {', '.join([p[0] for p in PROMUS])}")

    st.markdown("<br>", unsafe_allow_html=True)
    if calcul_autorise:
        run_sim = st.button("LANCER LA SIMULATION", use_container_width=True)
    else:
        run_sim = False

with col_affichage:
    st.markdown("## RÉSULTATS DE LA NOUVELLE SAISON")
    st.markdown("---")

    if run_sim or 'resultats' in st.session_state:
        if run_sim or 'resultats' not in st.session_state:
            with st.spinner("Exécution de la simulation de base et calcul des forces des promus..."):
                forces_base, avantage_base, index_base = executer_calcul_modele(p_18, p_19, p_20, p_21, p_22, p_23,
                                                                                p_24, p_25)

                # 1. Simulation de base pour calibrer les promus
                sim_base = Simulateur(forces_base, avantage_base, index_base)
                resultats_base = sim_base.simuler_monte_carlo(n_simulations)

                # 2. Construction des forces de la nouvelle saison
                forces_saison = {}
                avantage_saison = {}
                index_saison = {}
                numero = 0
                nom_de_index = {idx: nom for nom, idx in index_base.items()}

                # Transfert des équipes maintenues
                for ancien_idx, (att, defe) in forces_base.items():
                    nom = nom_de_index[ancien_idx]
                    if nom in RELEGUES:
                        continue
                    forces_saison[numero] = (att, defe)
                    avantage_saison[numero] = avantage_base[ancien_idx]
                    index_saison[nom] = numero
                    numero += 1

                # Injection des promus
                for nom, position_cible in PROMUS:
                    att, defe = forces_pour_position_cible(position_cible, resultats_base, forces_base, index_base)
                    forces_saison[numero] = (att, defe)
                    avantage_saison[numero] = np.mean(list(avantage_base.values()))
                    index_saison[nom] = numero
                    numero += 1

                # 3. Simulation finale de la nouvelle saison
                sim_nouvelle = Simulateur(forces_saison, avantage_saison, index_saison)
                st.session_state.resultats = sim_nouvelle.simuler_monte_carlo(n_simulations)
                st.session_state.sim_obj = sim_nouvelle
                st.session_state.forces_obj = forces_saison
                st.session_state.index_obj = index_saison

        res = st.session_state.resultats
        sim_obj = st.session_state.sim_obj
        forces_actives = st.session_state.forces_obj
        index_actif = st.session_state.index_obj
        visu = Visualiseur(res)

        # 1. MÉTRIQUES DE L'ÉQUIPE CIBLE
        st.markdown(f"### PARAMÈTRES ET PRÉVISIONS : {equipe_choisie.upper()}")

        idx_club = index_actif[equipe_choisie]
        att, deff = forces_actives[idx_club]
        stats = res.loc[equipe_choisie]

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Points attendus", f"{stats['points_moyens']:.1f}")
        col_m2.metric("Position moyenne", f"{stats['position_moyenne']:.1f}")
        col_m3.metric("Force d'attaque", f"{att:+.3f}")
        col_m4.metric("Force de défense", f"{deff:+.3f}")

        # 2. GRAPHIQUES
        st.markdown("<br>", unsafe_allow_html=True)
        col_g1, col_g2 = st.columns([1, 1], gap="medium")
        with col_g1:
            st.plotly_chart(visu.graphique_distribution_points(equipe_choisie, sim_obj), use_container_width=True)
        with col_g2:
            st.plotly_chart(visu.graphique_probabilites_titre(), use_container_width=True)

        st.plotly_chart(visu.graphique_classement_moyen(), use_container_width=True)

        # 3. MATRICE COMPLÈTE
        st.markdown("### DONNÉES BRUTES")
        df_propre = res.copy()
        df_propre['proba_titre'] = (df_propre['proba_titre'] * 100).round(1).astype(str) + " %"
        df_propre['proba_top4'] = (df_propre['proba_top4'] * 100).round(1).astype(str) + " %"
        df_propre['proba_relegation'] = (df_propre['proba_relegation'] * 100).round(1).astype(str) + " %"
        df_propre['position_moyenne'] = df_propre['position_moyenne'].round(1)
        df_propre['points_moyens'] = df_propre['points_moyens'].round(1)

        df_propre.columns = ["Points Moyens", "Position Moyenne", "Proba Titre", "Proba Top 4", "Proba Relégation"]
        st.dataframe(df_propre, use_container_width=True)

    else:
        st.write("Le système est en attente. Réglez les poids et lancez la simulation.")
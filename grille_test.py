import pandas as pd

EQUIPES = ["Equipe 1", "Equipe 2", "Equipe 3", "Equipe 4", "Equipe 5", "Equipe 6"]
#on chosit des noms d'équipes au piff

# --- Les sous-colonnes de chaque case ---
SOUS_COLONNES = ["journee", "buts_dom", "buts_ext", "resultat",
                 "pts_dom", "pts_ext", "diff_buts", "total_buts"]

MATCHS = [
    ("Equipe 1", "Equipe 2", 1.1, 2, 1),
    ("Equipe 3", "Equipe 4", 1.2, 0, 0),
    ("Equipe 5", "Equipe 6", 1.3, 3, 1),
    ("Equipe 1", "Equipe 3", 2.1, 1, 0),
    ("Equipe 2", "Equipe 5", 2.2, 2, 2),
    ("Equipe 4", "Equipe 6", 2.3, 1, 3),
    ("Equipe 1", "Equipe 4", 3.1, 4, 0),
    ("Equipe 3", "Equipe 6", 3.2, 1, 1),
    ("Equipe 5", "Equipe 2", 3.3, 0, 1),
    ("Equipe 1", "Equipe 5", 4.1, 2, 2),
    ("Equipe 6", "Equipe 2", 4.2, 0, 2),
    ("Equipe 4", "Equipe 3", 4.3, 1, 2),
    ("Equipe 1", "Equipe 6", 5.1, 3, 0),
    ("Equipe 2", "Equipe 4", 5.2, 1, 1),
    ("Equipe 3", "Equipe 5", 5.3, 2, 1),
    ("Equipe 2", "Equipe 3", 6.1, 0, 0),
    ("Equipe 6", "Equipe 1", 6.2, 1, 2),
    ("Equipe 4", "Equipe 5", 6.3, 2, 3),
]
######Exemple type pour avoir au moins des résultats

def construire_grille():
    """Construit et renvoie la grille MultiIndex (DataFrame Pandas).

    Colonnes : MultiIndex (equipe_a_domicile,  ajout des sous colonnes)
    Lignes   : equipe a l'exterieur,
    Chaque case (ext, dom) decrit un match ; la diagonale reste vide (NaN).
    """
    colonnes = pd.MultiIndex.from_product([EQUIPES, SOUS_COLONNES],
                                          names=["Domicile", "info"])
    grille = pd.DataFrame(index=EQUIPES, columns=colonnes, dtype=object)
    grille.index.name = "Exterieur"

    for dom, ext, journee, bd, be in MATCHS: #on prend MATCHS
        ####permet d'afficher le résduslat sous la forme V,N et D
        #### attention, ici on a une vison centré vers l'équipe qui jou eà domicile
        if bd > be:
            res, pts_dom, pts_ext = "V", 3, 0
        elif bd == be:
            res, pts_dom, pts_ext = "N", 1, 1
        else:
            res, pts_dom, pts_ext = "D", 0, 3
        valeurs = {
            "journee": journee, "buts_dom": bd, "buts_ext": be, "resultat": res,
            "pts_dom": pts_dom, "pts_ext": pts_ext,
            "diff_buts": bd - be, "total_buts": bd + be,
        }
        for col, val in valeurs.items():
            grille.loc[ext, (dom, col)] = val

    return grille


if __name__ == "__main__":
    grille = construire_grille()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)
    print("=== GRILLE (colonne = equipe domicile, ligne = equipe exterieur) ===\n")
    print(grille.fillna("---"))
    n = grille.xs("journee", axis=1, level="info").notna().sum().sum()
    print(f"\nNombre de matchs : {n}")


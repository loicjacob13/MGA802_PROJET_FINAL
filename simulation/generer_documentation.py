"""
generer_documentation.py — Script de génération de la documentation Sphinx
Auteurs : Fabien, Loïc, Guillaume — Projet MGA802 Groupe 2

UTILISATION :
    Depuis le terminal :
        python generer_documentation.py

DESCRIPTION :
    Ce script automatise la création de la documentation du projet.
    1. Vérifie la présence de Sphinx et l'installe si nécessaire.
    2. Crée un fichier de configuration (conf.py) s'il est manquant.
    3. Scanne les modules Python pour générer les sources (.rst).
    4. Compile le tout en un site web HTML.
"""

import os
import sys
import subprocess

# -----------------------------------------------------------------------
# 1. Configuration des chemins
# -----------------------------------------------------------------------
# On identifie la racine du projet, que le script soit lancé depuis "simulation" ou la racine
dossier_actuel = os.path.dirname(os.path.abspath(__file__))

if os.path.basename(dossier_actuel) == "simulation":
    racine_projet = os.path.dirname(dossier_actuel)
else:
    racine_projet = dossier_actuel

dossier_docs = os.path.join(racine_projet, "docs")
source_docs = os.path.join(dossier_docs, "source")
build_html = os.path.join(dossier_docs, "build", "html")


def afficher_etape(titre):
    """Affiche une séparation visuelle claire pour le terminal."""
    print(f"\n{'-' * 60}")
    print(f" {titre}")
    print(f"{'-' * 60}")


# -----------------------------------------------------------------------
# 2. Vérification et installation de Sphinx
# -----------------------------------------------------------------------
afficher_etape("ÉTAPE 1 — Vérification des dépendances (Sphinx)")
try:
    resultat = subprocess.run(
        [sys.executable, "-m", "sphinx", "--version"],
        capture_output=True, text=True, check=True
    )
    print(f"  Sphinx détecté : {resultat.stdout.strip()}")
except subprocess.CalledProcessError:
    print("  Sphinx introuvable. Installation en cours via pip...")
    subprocess.run([sys.executable, "-m", "pip", "install", "sphinx"], check=True)
    print("  Installation de Sphinx terminée.")

# -----------------------------------------------------------------------
# 3. Vérification du fichier de configuration (conf.py)
# -----------------------------------------------------------------------
fichier_conf = os.path.join(source_docs, "conf.py")

if not os.path.exists(fichier_conf):
    print("\n  [Info] Fichier conf.py manquant. Génération automatique...")
    os.makedirs(source_docs, exist_ok=True)

    with open(fichier_conf, "w", encoding="utf-8") as f:
        f.write("# Configuration générée automatiquement\n")
        f.write("import os\nimport sys\n")
        f.write("sys.path.insert(0, os.path.abspath('../..'))\n\n")

        f.write("project = 'Projet MGA802 - Simulation PL'\n")
        f.write("copyright = '2026, Fabien, Loïc, Guillaume'\n")
        f.write("author = 'Fabien, Loïc, Guillaume'\n\n")

        f.write("extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode']\n")
        f.write("html_theme = 'alabaster'\n")
    print("  Fichier conf.py créé avec les paramètres par défaut.")

# -----------------------------------------------------------------------
# 4. Génération des fichiers sources (.rst)
# -----------------------------------------------------------------------
afficher_etape("ÉTAPE 2 — Analyse du projet (sphinx-apidoc)")

# Fichiers et dossiers à exclure de la documentation
fichiers_a_ignorer = [
    "main_OLD.py", "main_streamlit.py", "main_NEW.py",
    "grille_test.py", "test_donnees.py", "requirement.py",
    "visualisation_streamlit.py", ".venv", "docs", "tests"
]
exclusions = [os.path.join(racine_projet, exc) for exc in fichiers_a_ignorer]

commande_apidoc = [
                      sys.executable, "-m", "sphinx.ext.apidoc",
                      "-o", source_docs,
                      "-f",  # Forcer l'écrasement des anciens fichiers
                      "-e",  # Un fichier .rst par module
                      "-M",  # Mettre les modules parents en premier
                      racine_projet
                  ] + exclusions

print("  Génération des fichiers .rst en cours...")
subprocess.run(commande_apidoc, capture_output=True, text=True, cwd=racine_projet)
print("  Fichiers .rst générés avec succès.")

# -----------------------------------------------------------------------
# 5. Compilation de la documentation HTML
# -----------------------------------------------------------------------
afficher_etape("ÉTAPE 3 — Construction de la documentation (sphinx-build)")

commande_build = [
    sys.executable, "-m", "sphinx",
    "-b", "html",
    source_docs,
    build_html,
    "-q",  # Mode silencieux pour la sortie standard
    "-E",  # Forcer une reconstruction complète
]

print("  Compilation des pages HTML...")
resultat_build = subprocess.run(
    commande_build,
    capture_output=True,
    text=True,
    cwd=racine_projet,
    env={**os.environ, "PYTHONPATH": racine_projet}
)

# Affichage des avertissements pertinents
if resultat_build.stderr:
    for ligne in resultat_build.stderr.splitlines():
        if ligne.strip() and not ligne.startswith("WARNING: autodoc"):
            print(f"  Avertissement: {ligne}")

# -----------------------------------------------------------------------
# 6. Conclusion et ouverture
# -----------------------------------------------------------------------
afficher_etape("RÉSULTAT")

index_html = os.path.join(build_html, "index.html")

if os.path.exists(index_html) and resultat_build.returncode == 0:
    print("  Documentation générée avec succès.")
    print(f"\n  Chemin du fichier d'index :")
    print(f"  {index_html}\n")

    # Commande suggérée pour ouvrir directement
    if sys.platform == "darwin":
        print(f"  Commande pour ouvrir : open '{index_html}'")
    elif sys.platform == "win32":
        print(f"  Commande pour ouvrir : start '{index_html}'")
else:
    print("  La génération a échoué.")
    print(f"  Code de retour de Sphinx : {resultat_build.returncode}")
    print("  Vérifiez les avertissements ci-dessus pour plus de détails.")
    sys.exit(1)
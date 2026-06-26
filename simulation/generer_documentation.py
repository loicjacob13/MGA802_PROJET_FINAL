"""
generer_documentation.py — Script principal de génération de la documentation Sphinx
Auteurs : Fabien - Loïc - Guillaume — Projet MGA802 Groupe 2

UTILISATION :
    Depuis la RACINE du projet (là où se trouvent donnees/, simulation/, etc.) :

        python generer_documentation.py

    La documentation HTML sera générée dans :
        docs/build/html/index.html

PRINCIPE :
    1. Sphinx lit conf.py pour connaître le projet et les extensions
    2. sphinx-apidoc génère un fichier .rst par module Python
    3. sphinx-build -b html construit les pages HTML depuis les .rst
    4. autodoc importe chaque module et lit les __doc__ de chaque fonction/classe
"""

import os
import sys
import subprocess


# -----------------------------------------------------------------------
# 1. Chemins
# -----------------------------------------------------------------------
RACINE_PROJET = os.path.dirname(os.path.abspath(__file__))
DOSSIER_DOCS  = os.path.join(RACINE_PROJET, "docs")
SOURCE        = os.path.join(DOSSIER_DOCS, "source")
BUILD_HTML    = os.path.join(DOSSIER_DOCS, "build", "html")


def afficher(message):
    """Affiche un message formaté dans le terminal."""
    print(f"\n{'='*60}")
    print(f"  {message}")
    print(f"{'='*60}")


# -----------------------------------------------------------------------
# 2. Vérification de Sphinx
# -----------------------------------------------------------------------
afficher("ÉTAPE 1 — Vérification de Sphinx")
try:
    resultat = subprocess.run(
        [sys.executable, "-m", "sphinx", "--version"],
        capture_output=True, text=True
    )
    print(f"  Sphinx trouvé : {resultat.stdout.strip()}")
except Exception:
    print("  Sphinx non trouvé. Installation en cours...")
    subprocess.run([sys.executable, "-m", "pip", "install", "sphinx"], check=True)
    print("  Sphinx installé.")


# -----------------------------------------------------------------------
# 3. sphinx-apidoc : génère un .rst par module Python
#    "Il nous faut un fichier .rst par fichier .py dans notre code"
#    "La commande sphinx-apidoc les génère automatiquement"
#    Commande : sphinx-apidoc -o source .
# -----------------------------------------------------------------------
afficher("ÉTAPE 2 — Génération des fichiers .rst (sphinx-apidoc)")

# Dossiers à documenter
modules_a_documenter = [
    os.path.join(RACINE_PROJET, "donnees"),
    os.path.join(RACINE_PROJET, "simulation"),
    os.path.join(RACINE_PROJET, "visualisation"),
    os.path.join(RACINE_PROJET, "controle_input.py"),
]

# Fichiers / dossiers à exclure de la génération automatique
exclusions = [
    os.path.join(RACINE_PROJET, "main_OLD.py"),
    os.path.join(RACINE_PROJET, "main_streamlit.py"),
    os.path.join(RACINE_PROJET, "main_NEW.py"),
    os.path.join(RACINE_PROJET, "grille_test.py"),
    os.path.join(RACINE_PROJET, "test_donnees.py"),
    os.path.join(RACINE_PROJET, "requirement.py"),
    os.path.join(RACINE_PROJET, "visualisation_streamlit.py"),
]

# On lance sphinx-apidoc pour chaque module séparément
# -o SOURCE  : dossier de sortie des .rst
# -f         : force l'écrasement des .rst existants
# -e         : un fichier .rst par module (recommandé)
# -M         : met le module parent en premier
for module in modules_a_documenter:
    if not os.path.exists(module):
        print(f" Module non trouvé (ignoré) : {module}")
        continue

    commande = [
        sys.executable, "-m", "sphinx.ext.apidoc",
        "-o", SOURCE,          # dossier de sortie : docs/source/
        "-f",                  # force l'écrasement
        "-e",                  # un .rst par module
        "-M",                  # module parent en premier
        module,                # chemin du module à documenter
    ] + exclusions

    print(f"  sphinx-apidoc → {os.path.basename(module)}")
    resultat = subprocess.run(commande, capture_output=True, text=True, cwd=RACINE_PROJET)
    if resultat.returncode != 0:
        print(f" Avertissement : {resultat.stderr[:200]}")
    else:
        print(f" Fichiers .rst générés")


# -----------------------------------------------------------------------
# 4. sphinx-build : construit la documentation HTML
#    Équivalent de : make html
# -----------------------------------------------------------------------
afficher("ÉTAPE 3 — Construction de la documentation HTML (sphinx-build)")

commande_build = [
    sys.executable, "-m", "sphinx",
    "-b", "html",              # format de sortie : HTML
    SOURCE,                    # dossier source (contient conf.py)
    BUILD_HTML,                # dossier de destination
    "-E",                      # force la reconstruction complète
    "-W", "--keep-going",      # traite les warnings mais continue
]

print(f"  Source : {SOURCE}")
print(f"  Sortie : {BUILD_HTML}")
print()

resultat_build = subprocess.run(
    commande_build,
    capture_output=True,
    text=True,
    cwd=RACINE_PROJET,
    env={**os.environ, "PYTHONPATH": RACINE_PROJET},  # pour que autodoc trouve les modules
)

# Afficher la sortie de Sphinx
if resultat_build.stdout:
    for ligne in resultat_build.stdout.splitlines():
        if ligne.strip():
            print(f"  {ligne}")

if resultat_build.stderr:
    for ligne in resultat_build.stderr.splitlines():
        if ligne.strip() and not ligne.startswith("WARNING: autodoc"):
            print(f"  ⚠  {ligne}")


# -----------------------------------------------------------------------
# 5. Résumé final
# -----------------------------------------------------------------------
afficher("RÉSULTAT")

index_html = os.path.join(BUILD_HTML, "index.html")
if os.path.exists(index_html):
    print(f" Documentation générée avec succès")
    print(f"\n  Ouvrir dans un navigateur :")
    print(f"  {index_html}")
    print()
    print("  Ou depuis le terminal :")
    if sys.platform == "darwin":
        print(f"  open '{index_html}'")
    elif sys.platform == "win32":
        print(f"  start '{index_html}'")
    else:
        print(f"  xdg-open '{index_html}'")
else:
    print(" La génération a échoué.")
    print(f"  Code de retour : {resultat_build.returncode}")
    print("  Vérifiez les messages d'erreur ci-dessus.")
    sys.exit(1)
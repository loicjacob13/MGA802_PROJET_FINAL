
import numpy as np
from grille_test import construire_grille
grille = construire_grille()




#on va commencer par inventer 3 équipes poour vérifier la voracité de notre modèle qu'on va ensuite entrrainer
# [index domicile (quel équipe joue au domicile), index extérieur (quel équipe joue extérieur], but par équipe domicile, buts par équipe extérieur

matchs_tets=np.array([0,1,3,1],
                     [1,2,0,0],
                     [2,0,1,2],
                     [0,2,2,0],
                     [1,0,1,1],
                     [2,1,0,3])
#modèle de poisson cakcule les buts attendus à partir de données
#faire une fonction de perte qui va calculer la perte


#idée 1 : faire un bvecteur qui stocke tout les infortmations nécessairees pour calculer landa et nhu avec formules mathématique s

#on part de 0 et on laisse le programme
# Convention : [attaque_0, attaque_1, attaque_2, defense_0, defense_1, defense_2, avantage]

nb_equipes=3 #on init au début 3 équipes uniquement

params=np.zeros(2*nb_equipes+1) #on fait exprés de partir de 0 pour qu'ensuite il prévoit quels infos remplir automatiquement
######### param est la donnée clé ##################



#affichage pour avoir accés aux nombre de buts pour chaque équipe

def deplier(params,n):
    attaques=params[0:n]
    defenses=params[n+1:2*n]
    avantage=params[2*n]
    return attaques,defenses,avantage

attaques,defenses,avantage=deplier(params,nb_equipes)
print




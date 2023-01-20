# Notes personnelles pour le WordEmbedding

## Etat des Lieux :

1ere Méthode : Latent Semantic Analysis [LSA](https://en.wikipedia.org/wiki/Latent_semantic_analysis)

Pour l'instant des résultats peu satisfaisant avec une méthode ne basant que sur la liste des mots dans le document => Pas de prise en compte de la position (et donc des mots adjacents)



Donnés chiffrées : 

Faut checkout sur la branche TF-IDF-Interface comme j'ai pas encore pull pour faire des tests, suffit de generer la WF matrix mais franchement l'interface est pas pratique... 

Faut que je regarde avec Majd la ou il en est.

## Objectif 

2ème Méthode : Proper Word Embedding 

J'ai recodé le tout mais franchement on va passer à la librairie suivante : [gensim](https://radimrehurek.com/gensim/). C'est prévu pour et c'est plus rapide askip

De plus, j'ai envie de passer à des méthodes + utilisées (les résultats de ce genre de méthode sont assez approximatifs => Pas le state of art pour une raison)

## Problèmes rencontrés

1. Coder un gros projet ca demande de setup un environnement de code fonctionnel et c'est parfois plus compliqué que prévu 
2. Faut réfléchir à une structure avant d'avancer comme un bourrin sinon on est pas efficace (cf j'ai 2 repo, des branches pas ouf et des issues alors que bon je code seul cette partie, je suis meme pas d'accord avec moi même sur la facon de proceder...)
3. Difficile de se concentrer pleinement sur le PSC en ne le travaillant que le mercredi => J'ai push plutot le mardi/mercredi, je fais plutot autre chose dans la semaine et c'est pas efficace meme si je prends plus de recul
4. Mauvaise communication dans l'équipe... Qui fait quoi ? On en est ou ? 
5. Problèmes de sommeil...
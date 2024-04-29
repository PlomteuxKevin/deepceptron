# deepceptron

## Objectif du projet
Afin d'aller plus loin dans la compréhension des réseaux de neurones, j'ai développé sur base du perceptron ma propre bibliothèque.
Le but était de comprendre le fonctionnement des différents éléments qui compose le deeplearning : Fonction d'activation, fonction de perte, backpropagation et métrics.

## Contenu
### Main file
**iris.py :** contient le model, charge le dataset iris.csv et prédit le résultat de classification.
**exam.py :** même exercice que iris.py mais sur d'autres données.

### Class
/akiplot/akiplot.py : une surclasse créé pour faciliter la création de graphique avec MathPlotLib et simplifier le code.
/pitch/perceptron.py : le perceptron en lui même.
/pitch/vprint.py : une surclasse à print() permettant de faciliter l'affichage en mode verbose du model.
/pitch/pitch_class.py : class de création du model incluant : préparation des données d'entrainement et de prédiction, création des graphs, entrainement, prédiction, verbose, etc.

## Résultat
Comparé au perceptron (image de supérieur) les résultats sur le dataset exam.csv sont plus précis et permettent une meilleur classification avec le deepceptron (image de droite).

Résultat avec le perceptron :
![perceptron](https://github.com/PlomteuxKevin/deepceptron/assets/168406292/13cec4bf-7320-448a-9658-34d2f5526629)

Résultat avec le deepceptron (réseau de neurone) :
![deepceptron](https://github.com/PlomteuxKevin/deepceptron/assets/168406292/6813a2bc-d249-4e11-b36c-58ca488b41a1)

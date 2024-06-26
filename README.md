# deepceptron

## Objectif du projet
Afin d'aller plus loin dans la compréhension des réseaux de neurones, j'ai développé sur base du perceptron ma propre bibliothèque.
Le but était de comprendre le fonctionnement des différents éléments qui compose le deeplearning : Fonction d'activation, fonction de perte, backpropagation et métrics.

## Contenu
### Main file
**exam.py :** contient le model, charge le dataset exam.csv et prédit le résultat de classification.<br />

### Class
**/akiplot/akiplot.py :** une surclasse créé pour faciliter la création de graphique avec MathPlotLib et simplifier le code.<br />
**/pitch/perceptron.py :** le perceptron en lui même.<br />
**/pitch/vprint.py :** une surclasse à print() permettant de faciliter l'affichage en mode verbose du model.<br />
**/pitch/pitch_class.py :** class de création du model incluant : préparation des données d'entrainement et de prédiction, création des graphs, entrainement, prédiction, verbose, etc.<br />
**/pitch/layer_class.py :** class permettant la mise en couche (layer) des perceptrons afin de créer un réseau de neurone.

## Résultat
Comparé au perceptron, les résultats sur le dataset exam.csv du deepceptron sont plus précis et permettent une meilleur classification non linéaire.

Résultat avec le perceptron : <br />
![perceptron](https://github.com/PlomteuxKevin/deepceptron/assets/168406292/13cec4bf-7320-448a-9658-34d2f5526629)

Résultat avec le deepceptron (réseau de neurone) : <br />
![deepceptron](https://github.com/PlomteuxKevin/deepceptron/assets/168406292/6813a2bc-d249-4e11-b36c-58ca488b41a1)

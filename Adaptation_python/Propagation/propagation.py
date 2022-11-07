'''Création d'un réseau de neurones :
Adaptation en Python.
Propagation des données à l'intérieur du réseau de neurones, de l'input à l'output.

Source : 
HEUDIN J.-C., Comprendre le Deep Learning, une introduction aux réseaux de neurones, Science-eBook, Paris, 2016, ISBN:979-1091245449, p.87-117.
'''

import math

# trois couches de neurones
Input = [0, 0, 0, 0]
Hidden = [0, 0, 0, 0]
Output = [0, 0]

# poids synaptiques
Wh = [[0.5, 0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5, 0.5]];
# poids des connexions de la couche cachée à la couche de sortie (hidden => output)
Wo = [[0.5, 0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5, 0.5]];

# données d'entrées 
input_data = [0, 1, 0, 1]



# fonction d'activation (sigmoïde)
def sigmoid(x):
    return 1/(1 + math.pow(math.e, (-1 * x)) )

# propaation des données (input => output)
def propagate(d):
    global Input, Hidden, Output, Wh, Wo
    for x in range(len(Input)) : 
        Input[x] = d[x]

    Xh = [0, 0, 0, 0]

    # calcul de la somme pondérée avec une double itération
    for j in range(len(Hidden)):
        for i in range(len(Input)) :
            Xh[j] += Wh[j][i] * Input[i]


    # fonction d'activation sur les valeurs de Xh
    for j in range(len(Hidden)) :
        Hidden[j] = sigmoid(Xh[j])

    # propagation vers la couche de sortie 
    Xo = [0, 0]
    for k in range(len(Output)) :
        for j in range(len(Hidden)) :
            Xo[k] += Wo[k][j] * Hidden[j]

    # fonction d'activation sur les valeurs de Xo
    for k in range(len(Output)) :
        Output[k] = sigmoid(Xo[k]) 
        
    return Output



### Appel de la fonction : 
propagate(input_data)

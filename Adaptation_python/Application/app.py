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

# taux d'apprentissage
alpha = 0.5


# données d'entrées 
input_data = [1, 0, 1, 0]


# liste de données que l'on souhaite obtenir en sortie
Target = [1, 0]


# fonction d'activation (sigmoïde)
def sigmoid(x):
    return 1/(1 + math.pow(math.e, (-1 * x)) )

# propagation des données (input => output)
def propagate(d):
    global Input, Hidden, Output, Wh, Wo
    for x in range(len(Input)) : 
        Input[x] = d[x]

    # propagation dans la couche cachée
    Xh = [0, 0, 0, 0]
    # calcul de la somme pondérée avec une double itération
    for j in range(len(Hidden)):
        for i in range(len(Input)): 
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

# liste contenant les erreurs (différence entre erreur obtenue - erreur souhaitée)
Err = []

# fonction d'apprentissage
def learn(d): 

    # calcul de l'erreur : Target - Output
    for k in range(len(Output)):
        err = Target[k] - Output[k]
        # on ajoute la différence à la liste
        Err.append(err)


    # calculer les gradients d'erreurs de la couche de sortie
    Wog = [[0, 0, 0, 0], [0, 0, 0, 0]] 

    for k in range(len(Output)):
        for j in range(len(Hidden)):
            Wog[k][j] = -Err[k] * Output[k] * (1 - Output[k]) * Hidden[j]

    
    # calculer les gradients d'erreur de la couche cachée
    Whg = [[0, 0, 0, 0], [0, 0, 0, 0],
            [0, 0, 0, 0],[0, 0, 0, 0]]

    for j in range(len(Hidden)):
        for i in range(len(Input)):
            e = 0
            for k in range(len(Output)):
                e += Wo[k][j] * Err[k]
            Whg[j][i] = -e * Hidden[j] * (1 - Hidden[j]) * Input[i]

    # mise à jour des poids de sortie
    for k in range(len(Output)):
        for j in range(len(Hidden)):
            Wo[k][j] -= alpha * Wog[k][j]

    # mise à jour des poids de la couche cachée
    for j in range(len(Hidden)):
        for i in range(len(Input)):
            Wh[j][i] -= alpha * Whg[j][i]

    
    return Output


def resultat():
    print("----------------------------")
    print("Sortie 1 : ", Output[0])
    print("Erreur 1 : ", Target[0] - Output[0])
    print("Sortie 2 : ", Output[1])
    print("Erreur 2 : ", Target[1] - Output[1])
    print("----------------------------")
    print("Valeur attendue : ", Target)
    print("Valeur obtenue : ", Output)

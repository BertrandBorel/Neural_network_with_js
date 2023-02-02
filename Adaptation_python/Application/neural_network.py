import numpy as np


# trois couches de neurones
Input = np.array([0, 0, 0, 0])
Hidden = np.array([0, 0, 0, 0])
Output = np.array([0, 0])

# poids synaptiques
Wh  = np.array([[0.5 for j in range(4)] for i in range(4)])

# poids des connexions de la couche cachée à la couche de sortie (hidden => output)
Wo = np.array([[0.5 for j in range(4)] for i in range(2)])

# taux d'apprentissage
alpha = 0.5

# données d'entrées 
input_data = np.array([1, 0, 1, 0])

# liste de données que l'on souhaite obtenir en sortie
Target = np.array([1, 0])


# fonction d'activation (sigmoïde)
def sigmoid(x):
    return 1/(1 + np.exp(-1 * x)) 


# propagation des données (input => output)
def propagate(d):
    global Input, Hidden, Output, Wh, Wo
    Input = d
    Xh = np.dot(Wh, Input)
    Hidden = sigmoid(Xh)
    Xo = np.dot(Wo, Hidden)
    Output = sigmoid(Xo)
    return Output


# liste contenant les erreurs (différence entre erreur obtenue - erreur souhaitée)
Err = []
Err = np.array(Err)


# fonction d'apprentissage
def learn(d): 
    global Wh, Wo

    # calcul de l'erreur : Target - Output
    Err = Target - Output

    # calculer les gradients d'erreurs de la couche de sortie
    Wog = np.array([[0 for j in range(4)] for i in range(2)])
    Wog = -np.outer(Err * Output * (1 - Output), Hidden)

    # calculer les gradients d'erreur de la couche cachée
    Whg = np.array([[0 for j in range(4)] for i in range(4)])
    Whg = -Err @ Wo * Hidden * (1 - Hidden) * Input.T

    # mise à jour des poids de sortie
    Wo = Wo - alpha * Wog

    # mise à jour des poids de la couche cachée
    Wh = Wh - alpha * Whg
    
    return Output



# Affichage simplifié des résultats
def resultat_simple():
        print("Valeur obtenue : ", Output, end="\n")
        # affichage des erreurs   
        erreurs = Output - Target
        print("-----> Erreurs : ", erreurs[0], " ", erreurs[1], end="\n") 

# lancement de l'application : propagation, apprentissage, résultat.       
def application():
    propagate(input_data)
    learn(input_data)
    resultat_simple()



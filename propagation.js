// Création d'un réseau de neurones avec Javascript :
// Propagation des données à l'intérieur du réseau de neurones, de l'input à l'output.

// Source : 
// HEUDIN J.-C., Comprendre le Deep Learning, une introduction aux réseaux de neurones, Science-eBook, Paris, 2016, ISBN:979-1091245449, p.87-117.


// trois couches de neurones
var Input = [];
var Hidden = [];
var Output = [];

// poids synaptiques
var Wh = [];
var Wo = [];

// fonction d'initialisation des différents tableaux :
// initialisation des valeurs des neurones de chaque couche :
// 4 pour input, 4 pour hidden, 2 pour output.
// + poids initialisés à 0.5
function reset () {
    Input = [0, 0, 0, 0]
    Hidden = [0, 0, 0, 0]
    Output = [0, 0, 0, 0]

// coefficients synaptiques des liens reliant les neurones entre eux (input => hidden)
// chaque neurones en input est connecté à tous les neurones hidden
    Wh = [[0.5, 0.5, 0.5, 0.5],
          [0.5, 0.5, 0.5, 0.5],
          [0.5, 0.5, 0.5, 0.5],
          [0.5, 0.5, 0.5, 0.5]];

// poids des connexions de la couche cachée à la couche de sortie (hidden => output)
    Wo = [[0.5, 0.5, 0.5, 0.5],
          [0.5, 0.5, 0.5, 0.5]];

}

// Données présentées au réseau 
var input_data = [0, 1, 0, 1];

// fonction d'activation : sigmoïde
// permet de calculer la valeur de sortie d'un neurone
// utilisée par fonction propagate()
// appel à la librairie Math pour puissance (pow) et constante (E)
function sigmoid (x) {
    return 1 / (1 + Math.pow(Math.E, (-1 * x)));
}

// propagate () : propage les données de l'input vers l'output
// commence par copier les données d'un tableau (d) passé en argument dans la couche d'entrée
function propagate (d) {
    // copie les données de l'input
    for (var i = 0; i < Input.length; i++) {
        Input[i] = d[i];
    }

    // propagation dans la couche cachée (hidden)
    // initialisation des valeurs 
    Xh = [0, 0, 0, 0];
    // calcul de la somme pondérée avec une double itération
    // pour chaque neurone en hidden, on calcule la somme pondérée de la valeur de chaque neurone en input
    // ... multipliée par le poids synaptique du lien associé 
    for (var j = 0; j < Hidden.length; j++) {
        for (var i = 0; i < Input.length; i++) {
            Xh[j] += Wh[j][i] * Input[i];
            // Xh contient alors les sommes cumulées pour la couche hidden 
        }
    }

    // application de la fonction d'activation sur les valeurs de Xh (hidden)
    for (var j = 0; j < Hidden.length; j++) {
        Hidden[j] = sigmoid(Xh[j]);
    }

    // propagation dans la couche de sortie (hidden vers output)
    // même principe : calcul des sommes pondérées par une double itération
    Xo = [0, 0];
    for (var k = 0; k < Output.length; k++) {
        for (var j = 0; j < Hidden.length; j++) {
            Xo[k] += Wo[k][j] * Hidden[j];
        }
    }
    
    // application de la fonction d'activation sur les valeurs de Xo (sortie)
    for (var k = 0; k < Output.length; k++) {
        Output[k] = sigmoid(Xo[k]);
        // les neurones de sortie contiennent les valeurs propagées depuis la couche d'entrée.
    }
}
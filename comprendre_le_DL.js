// Création d'un réseau de neurones avec Javascript
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
    Xh = [0, 0, 0, 0];
    for (var j = 0; j < Hidden.length; j++) {
        for (var i = 0; i < Input.length; i++) {
            Xh[j] += Wh[j][i] * Input[i];
        }
    }

    // application de la fonction d'activation
    for (var j = 0; j < Hidden.length; j++) {
        Hidden[j] = sigmoid(Xh[j]);
    }

    // propagation dans la couche de sortie 
    Xo = [0, 0];
    for (var k = 0; k < Output.length; k++) {
        for (var j = 0; j < Hidden.length; j++) {
            Xo[k] += Wo[k][j] * Hidden[j];
        }
    }
    
    // application de la fonction d'activation
    for (var k = 0; k < Output.length; k++) {
        Output[k] = sigmoid(Xo[k]);
    }
}
// Création d'un réseau de neurones avec Javascript :
// Apprentissage du réseau : algorithme de rétropropagation du gradient de l'erreur.
// Ajout d'une fonction learn(). 

// Source : 
// HEUDIN J.-C., Comprendre le Deep Learning, une introduction aux réseaux de neurones, Science-eBook, Paris, 2016, ISBN:979-1091245449, p.87-117.


// trois couches de neurones
var Input = [];
var Hidden = [];
var Output = [];

// poids synaptiques
var Wh = [];
var Wo = [];

// taux d'apprentissage
var alpha = 0.5

// tableau de données que l'on souhaite obtenir en sortie
var Target = [0, 0]

// fonction d'initialisation des différents tableaux :
// initialisation des valeurs des neurones de chaque couche :
// 4 pour input, 4 pour hidden, 2 pour output.
// + poids initialisés à 0.5
function reset () {
    Input = [0, 0, 0, 0];
    Hidden = [0, 0, 0, 0];
    Output = [0, 0];

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

// algorithme de rétropropagation :
// 4 étapes :
//     - le calcul de l'erreur en sortie après la propagation des données 
//     - le calcul des gradients d'erreurs pour corriger les poids synaptiques des neurones de la couche de sortie
//     - le calcul des gradients d'erreurs pour corriger les poids synaptiques des neurones de la couche cachée
//     - la mise à jour des poids synaptiques de la couche de sortie et de la couche cachée.

function learn () {
    
    // calcul de l'erreur : Target - Output
    var Err = [];
    for (var k = 0; k < Output.length; k++) {
        Err[k] = Target[k] - Output[k];
    }

    // calculer les gradients d'erreurs de la couche de sortie
    var Wog = [[0, 0, 0, 0], [0, 0, 0, 0]] ;

    for (var k = 0; k < Output.length; k++) {
        for (var j = 0; j < Hidden.length; j++) {
            Wog[k][j] = -Err[k] * Output[k] * (1 - Output[k]) * Hidden[j];
        }
    }

    // calculer les gradients d'erreurs de la couche cachée
    // boucle for supplémentaire pour rétropropager l'erreur de sortie vers hidden pour l'affecter aux différents neurones
    // ... en fonction de leurs poids synaptiques
    var Whg = [[0, 0, 0, 0], [0, 0, 0, 0],
            [0, 0, 0, 0],[0, 0, 0, 0]];

    for (var j = 0; j < Hidden.length; j++) {
        for (var i = 0; i < Input.length; i++) {
            var e = 0;
            for (var k = 0; k < Output.length; k++)
                e += Wo[k][j] * Err[k];
            Whg[j][i] = -e * Hidden[j] * (1 - Hidden[j]) * Input[i];

        }
    }

    // mise à jour de l'ensemble des poids synaptiques du réseau 
    // Chaque poids est modifié en lui soustrayant une portion du gradient d'erreur 
    // .... par l'application du taux d'apprentissage alpha.
    // Double itération : sur les poids de output et sur ceux de hidden

    // mise à jour des poids de sortie
    for (var k = 0; k < Output.length; k++) {
        for (var j = 0; j < Hidden.length; j++) {
            Wo[k][j] -= alpha * Wog[k][j];
        }
    }

    // mise à jour des poids de la couche cachée
    for (var j = 0; j < Hidden.length; j++) {
        for (var i = 0; i < Input.length; i++) {
            Wh[j][i] -= alpha * Whg[j][i];
        }
    }
}
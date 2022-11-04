// Création d'un réseau de neurones avec Javascript
// Source : 
// HEUDIN J.-C., Comprendre le Deep Learning, une introduction aux réseaux de neurones, Science-eBook, 2016, ISBN:979-1091245449, p.87-117.


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
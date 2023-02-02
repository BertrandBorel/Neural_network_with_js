import neural_network as NN
import time

# Définir le nombre d'itérations
iterations = 20

# Messages : nombre d'itérations et les valeurs attendues
print("Nombre d'itérations :", iterations)
print("\nValeur attendue : ", NN.Target, end="\n-----------\n")    

# Apprentissage
for i in range(iterations):
    print("Epoch ", i, ":")
    NN.application()
    time.sleep(1)
        
# Message : fin du programme        
print("\n---------\nFin du programme.")
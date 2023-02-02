import programme as pgm

# Définir le nombre d'itérations
iterations = 20

# Messages : nombre d'itérations et les valeurs attendues
print("Nombre d'itérations :", iterations)
print("\nValeur attendue : ", pgm.Target, end="\n-----------\n")    

# Apprentissage
for i in range(iterations):
    print("Epoch ", i, ":")
    pgm.application()
    pgm.time.sleep(1)
        
# Message : fin du programme        
print("\n---------\nFin du programme.")

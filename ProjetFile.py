






#projet


import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import random
import matplotlib.animation as animation
from matplotlib import pyplot as plt

#Fonction de HIMMELBLAU
def f(x, y):
    return (x **2+ y - 11)**2 + (x + y**2 - 7)**2

"""
#FONCTIONS UTILISEES POUR LE TEST DE VALIDATION
#Fonction de RASTRIGIN
def f(x,y):
    return 20 + x * x + y * y - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))

#Fonction de BOOTH
def f(x,y):
    return (x+2*y-7)**2+(2*x+y-5)**2
"""

Nmax = 100
nbparticules = 200
ndim = 2
phi1 = 0.1
phi2 = 0.9
w=0.4

def PSO():
    """
    INITIALISATION
    """
    #X et V sont des tableaux à trois dimensions, vides dans un premier temps
    #X[i] contient la position des points (X[i] = (x[i],y[i]) au temps i
    #V[i] contient la vitesse au temps i
    X = np.empty((Nmax + 1, nbparticules, ndim))
    V = np.empty((Nmax + 1, nbparticules, ndim))

    #le premier élement de X contiendra nparticules points (x,y) pris au hasard compris entre -5 et 5
    X[0] = np.random.uniform(-5, 5, (nbparticules, 2))
    #le premier élement de V est un tableau contentant des zeros de dimension nbparticules x 2
    V[0] = np.zeros((nbparticules, 2))

    #pbest_X = la meilleure position de la particule
    #pbest = f(pbest_X), c'est le "personnal best"
    #On les initialise avec la position de départ
    pbest = f(X[0][0][0], X[0][0][1])
    pbest_X = np.array(X[0][0][0], X[0][1][1])

    #best_X est la meilleure position de toutes les particules
    #gbest = f(best_X), c'est le "global best"
    best_X = np.array([X[0][0][0], X[0][0][1]])
    gbest = f(X[0][0][0], X[0][0][1])


    """
    ALGORITHME
    """
    for k in range(0,Nmax):
        for i in range(0, nbparticules):
            U1 = random.uniform(0, 1)
            U2 = random.uniform(0, 1)
            newV = w*V[k][i] + phi1 * U1 * (pbest_X - X[k][i]) + phi2 * U2 * (best_X - X[k][i])
            newX = X[k][i] + newV
            V[k + 1][i] = newV
            X[k + 1][i] = newX
            if (f(newX[0], newX[1]) < gbest):
                gbest = f(newX[0], newX[1])
                best_X = newX
                pbest = f(newX[0], newX[1])
                pbest_X = np.array(newX[0], newX[1])
    print('Minimum = ', best_X)
    return X, best_X

X,best_X = PSO()

#On calcule l'erreur qui est la différence entre le minimum théorique
# et le minimum de que l'on a obtenu (position du "global best")
def erreur(best_X):
    Erreur = []
    if (best_X[0] < 0 and best_X[1] > 0):
        Erreur.append(abs(-2.805118 - best_X[0]))
        Erreur.append(abs(3.131312 - best_X[1]))
    if (best_X[0] < 0 and best_X[1] < 0):
        Erreur.append(abs(-3.779310 - best_X[0]))
        Erreur.append(abs(-3.283186 - best_X[1]))
    if (best_X[0] > 0 and best_X[1] < 0):
        Erreur.append(abs(3.584428 - best_X[0]))
        Erreur.append(abs(-1.848126 - best_X[1]))
    if (best_X[0] > 0 and best_X[1] > 0):
        Erreur.append(abs(3.0 - best_X[0]))
        Erreur.append(abs(2.0 - best_X[1]))
    return Erreur

erreur=erreur(best_X)

"""
MODELISATION
On modélise la convergence des particules avec un graphique animé
"""
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True, linestyle='-', color='0.75')
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
scat = plt.scatter(X[0][:, 0], X[0][:, 1])
Y, Z = np.meshgrid([-5, 5], [-5, 5])

def anime(i):
    scat.set_offsets(X[i])
    return scat,

anim = animation.FuncAnimation(fig, anime, frames=np.arange(0, Nmax, 1),
                              interval=100,repeat=False)

plt.title("Convergence des particules vers un minimum")

plt.text(1.2,3.2," w = 0.4\n phi1 = 1\n phi2 = 0.9 ")

plt.annotate(best_X,
         xy=(best_X[0],best_X[1]), xycoords='data',
         xytext=(+0, +15), textcoords='offset points', fontsize=7,
         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.annotate(erreur,
         xy=(erreur[0],erreur[1]), xycoords='data',
         xytext=(+2, +10), textcoords='offset points', fontsize=7)

plt.show()
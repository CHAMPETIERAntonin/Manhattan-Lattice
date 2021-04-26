import numpy as np
import math

###############################################
D = 0.5
v = 2
xmax = 6
dx = 0.01
tf = 1.3
dt = 0.01
mu = (D*dt) / dx ** 2
Lambda = (v * dt) / dx
Nt = int(tf/dt)
Nx = int(xmax * 2 / dx)

x = np.linspace(-xmax,xmax,Nx)

#################################################

def cree_matrice():
    M = np.array([[0.0 for i in range(Nx)] for j in range(Nx)])
    j = 0  # position du 1+2mu
    for i in range(Nx):
        if i == 0:
            M[0, 0] = 1 + 2 * mu
            M[0, 1] = - mu
        elif i == Nx - 1:
            M[i, i - 1] = -mu
            M[i, i] = 1 + 2 * mu
        else:
            M[i, j] = 1 + 2 * mu
            M[i, j - 1] = -mu
            M[i, j + 1] = -mu
        j += 1
    return M

M = cree_matrice()
inv_M = np.linalg.inv(M)

######################################################################

def carre_initial():
    valeur_init = np.array([[0]])
    for i in range(1,len(x)):
        if abs(x[i]) <= 1:
            valeur_init = np.append(valeur_init,[[1]],axis=0)
        else:
            valeur_init = np.append(valeur_init,[[0]],axis=0)
    return valeur_init

def carre_analytique(t):
    valeur = np.array([[0.5 * (math.erf((1-x[0])/math.sqrt(4*D*t)) + math.erf((1+x[0])/math.sqrt(4*D*t)))]])
    assert t != 0
    for i in range(1,len(x)):
        temp = [[0.5 * (math.erf((1-(x[i]-v*t))/math.sqrt(4*D*t)) + math.erf((1+(x[i]-v*t))/math.sqrt(4*D*t)))]]
        valeur = np.append(valeur,temp,axis=0)
    return valeur

def dirac_initial():
    valeur_init = np.array([[0]])
    for i in range(1,len(x)):
       if i == len(x) // 2:
            valeur_init = np.append(valeur_init,[[1]],axis=0)
       else:
            valeur_init = np.append(valeur_init,[[0]],axis=0)
    return valeur_init

def dirac_analytique(t):
    assert t != 0
    valeur = np.array([[1 / math.sqrt(4 * math.pi * D * t) * math.exp((-x[0] ** 2) / (4 * D * t))]])
    for i in range(1,len(x)):
        temp = 1 / math.sqrt(4 * math.pi * D * t) * math.exp((-(x[i] - v * t) ** 2) / (4 * D * t))
        valeur = np.append(valeur,[[temp]],axis=0)
    return valeur

def solution_numerique(u0):
    utemp = np.array([[(1-Lambda) * u0[0,0]]])
    for i in range(1,u0.shape[0]):
        utemp = np.append(utemp,[[(1 - Lambda) * u0[i,0] + Lambda * u0[i-1,0]]],axis=0)
    u1 = np.dot(inv_M,utemp)
    return u1

def condition_initiale(initiale):
    if initiale == "carre":
        u = carre_initial()
    if initiale == "dirac":
        u = dirac_initial()
    return u

###################################################################

def erreur(u_num,u_ana,Nx):
    dx = 2 * xmax / Nx
    erreur=0
    for i in range(Nx):
        temp=((u_num[i]-u_ana[i])**2)*dx
        erreur+=temp
    erreur=np.sqrt(erreur)
    return erreur

#######################################################

def regression_lineaire(x,y):
    S = len(x)
    Sx = 0
    for i in range(len(x)):
        Sx += x[i]
    Sy = 0
    for j in range(len(y)):
        Sy += y[j]
    Sxx = 0
    for k in range(len(x)):
        Sxx += x[k]**2
    Sxy = 0
    for l in range(len(x)):
        Sxy += x[l]*y[l]
    det = S*Sxx - Sx**2
    a = (S*Sxy - Sx*Sy)/det
    b = (Sxx*Sy - Sx*Sxy)/det
    a=np.floor((a*1000))/1000
    b=np.floor((b*1000))/1000
    return a,b

from Méthode import *
import numpy as np
import matplotlib.pyplot as plt
import math

################################################

initialisation = "carre"
#initialisation = "dirac"

trace = True
#trace = False

mesure_erreur = True
#mesure_erreur = False
val_erreur = 0


u0 = condition_initiale(initialisation)
plt.plot(x, u0[:,0],label="solution numerique")
plt.title("Resolution de l'équation de Fokker-Plank t = 0 s")
plt.legend()
plt.show()

if trace == True:
    for j in range(1,Nt+1):
        u1 = solution_numerique(u0)
        u0 = u1

        if j == 1:
            x_trace = [x[15 * i] for i in range(len(x) // 15 - 1)]
            u0_trace = [u0[15 * i, 0] for i in range(len(x) // 15 - 1)]
            #line, = plt.plot(x, u0[:, 0], "o", color="red", label="solution numerique")
            line, = plt.plot(x_trace, u0_trace, "o", color="red", label="solution numerique")
            if initialisation == "carre":
                line2, = plt.plot(x, carre_analytique(dt), label="solution analytique")
            elif initialisation == "dirac":
                line2, = plt.plot(x, dirac_analytique(dt), label="solution analytique")

        else:
            u0_trace = [u0[15*i,0] for i in range(len(x)//15-1)]
#            line.set_ydata(u0[:,0])
            line.set_ydata(u0_trace)
            if initialisation == "carre":
                line2.set_ydata(carre_analytique(j*dt)[:,0])
            elif initialisation == "dirac":
                line2.set_ydata(dirac_analytique(j*dt)[:,0])

        plt.pause(0.01)
        plt.xlabel("x")
        plt.ylabel("u(x)")
        instant = int(dt * j * 10) / 10
        plt.title("Resolution de l'équation de Fokker-Plank \n t = "+str(instant)+" s")
        plt.legend()

        if mesure_erreur == True and j == Nt:
            val_erreur = erreur(u0[:,0],carre_analytique(dt*j)[:,0],Nx)

    print(val_erreur)


plt.show()

tab_dt = [2**(-i) for i in range(3,8)]
tab_valeur =[0.2200586273302155,0.09174098937465444,0.03948221060137927,0.01643585075612612,0.006610093036526104]
plt.plot(np.log(tab_dt),-1*np.log(tab_valeur),"o")

a = -regression_linéaire(np.log(tab_dt),np.log(tab_valeur))[0]
b = -regression_linéaire(np.log(tab_dt),np.log(tab_valeur))[1]
x2=np.linspace(2**(-7),2**(-3),100)
plt.plot(np.log(x2),a*np.log(x2)+b, label=str(a) + "x + " + str(b))
plt.title("Evolution de l'erreur")
print(a,b)
plt.legend()
plt.show()

import math
import sys

__author__ = 'Antonin'
__Filename = 'ElephantWalk'
__Creationdate__ = '18/02/2021'

import random as rand
import matplotlib.pyplot as plt
from typing import Any, List, Callable, Optional
from Serious.Lattices import Lattice, OrderedLattice, UnOrderedLattice, UnOrderedLattice1D
from Serious.Lattices import Direction
import numpy as np
import seaborn as sn


# def proba(proba, rng: rand.Random = rand):
#     return 1 if rng.random() < proba else -1

def probab(proba, rng: rand.Random = rand):
    return rng.random() < proba

# class ElephantWalk1D:
#
#     def __init__(self, p, q):
#         self.p = p
#         self.q = q
#         self.listX = []
#
#
#     def walk(self, stepNumber):
#         x = 0
#         self.listX = [x]
#
#         listSigma = [proba(self.q)]
#         x += listSigma[0]
#         self.listX.append(x)
#
#         for i in range(stepNumber):
#             t = rand.randrange(len(listSigma))
#             sigma = proba(self.p) * listSigma[t]
#             x += sigma
#             self.listX.append(x)
#             listSigma.append(sigma)
#
#
#     def draw(self):
#         plt.plot(range(len(self.listX)), self.listX)
#         plt.grid()
#         plt.show()


def sign(a):
    return 1 if a > 0 else 0 if a == 0 else -1


class Walk:
    def __init__(self:'Walk', lattice: Lattice, seed: Any):
        self.seed = seed
        self.rand = rand.Random(self.seed)

        self.lattice = lattice

    def walk(self: 'Walk', stepNumber: int) -> None:
        return

    def draw(self:'Walk') -> None:
        return

    def calculateVariance(self:'Walk') -> Any:
        return []

    def calculateTheoricalVariance(self:'Walk') -> Optional[List[float]]:
        return []

    def _calculateListVariance(self:'Walk', liste:List[float]) -> List[float]:
        variances: List[float] = []
        for i in range(len(liste)):
            moyenne = sum(liste[:i+1]) / (i+1)
            print(moyenne)
            differenceSum = 0
            for pos in liste[:i+1]:
                differenceSum += (moyenne - pos)*(moyenne - pos)

            variances.append(differenceSum/(i+1))
        return variances



class ElephantWalk(Walk):

    # p la probabilité de d'aller dans le même sens
    # q la probabilité de départ de p
    # r la probabilité d'aller dans la même direction
    # s la probabilité  de départ de r

    def __init__(self: 'ElephantWalk', p: float, q: float, r: float, s: float, lattice: Lattice, seed: Any):
        """
        :param float p : la probabilité de rester dans le même sens
        :param float q : la probabilité de départ de p
        :param float r : la probabilité d'aller dans la même direction
        :param float s : la probabilité  de départ de r
        """

        super().__init__(lattice, seed)
        self.p = p
        self.q = q
        self.r = r
        self.s = s
        self.listX = []
        self.listY = []




    def walk(self: 'ElephantWalk', stepNumber: int) -> None:
        x = 0
        y = 0

        self.listX = [x]
        self.listY = [y]

        #listSigma = [proba(self.q) * (proba(self.s) + 3 )/2]           # signe pour le sens, nombre pour la direction
                                                                        # 1/-1 selon x, 2/-2 pour y
        direc = Direction.NORTH
        if(not probab(self.s, self.rand)):
            direc = direc.changeDirection()
        if(not probab(self.q, self.rand)):
            direc = direc.reverse()

        listeAvailableDirections = self.lattice.getAvailableDirections(0, 0)


        # if(not direc in listeAvailableDirections):
        #     d = direc
        #     for direcF in listeAvailableDirections:
        #         if direc.isSameDirection(direcF):
        #             d = direcF
        #     if(direc is d):
        #         for direcF in listeAvailableDirections:
        #             if direc.isSameSide(direcF):
        #                 direc = direcF
        #     if(direc is d):
        #             direc = listeAvailableDirections[0]

        if(not direc in listeAvailableDirections):
            if(direc.reverse() in listeAvailableDirections):
                direc = direc.reverse()
            elif(direc.changeDirection() in listeAvailableDirections):
                direc = direc.changeDirection()
            else:
                direc = listeAvailableDirections[0]


        listSigma = [direc]

        if direc.isHorizontal:
            x += direc.offset
        else:
            y += direc.offset

        self.listX.append(x)
        self.listY.append(y)

        for i in range(stepNumber):
            sigmaChosen = listSigma[rand.randrange(len(listSigma))]

            #sigma = proba(self.p) * (changeDirection[sigmaChosen] if proba(self.r) == -1 else sigmaChosen);

            if(not probab(self.p, self.rand)):
                sigmaChosen = sigmaChosen.reverse()
            if(not probab(self.r, self.rand)):
                sigmaChosen = sigmaChosen.changeDirection()

            listeAvailableDirections = self.lattice.getAvailableDirections(x, y)

            if(not sigmaChosen in listeAvailableDirections):
                if(sigmaChosen.reverse() in listeAvailableDirections):
                    sigmaChosen = sigmaChosen.reverse()
                elif(sigmaChosen.changeDirection() in listeAvailableDirections):
                    sigmaChosen = sigmaChosen.changeDirection()
                else:
                    sigmaChosen = listeAvailableDirections[0]

            # if(not sigmaChosen in listeAvailableDirections):
            #     d = sigmaChosen
            #     for direcF in listeAvailableDirections:
            #         if sigmaChosen.isSameDirection(direcF):
            #             sigmaChosen = direcF
            #     if(sigmaChosen is d):
            #         for direcF in listeAvailableDirections:
            #             if(sigmaChosen.isSameSide(direcF)):
            #                 sigmaChosen = direcF
            #     if(sigmaChosen is d):
            #         sigmaChosen = listeAvailableDirections[0]
            #     # else:
            #     #     sigmaChosen = d


            if sigmaChosen.isHorizontal:
                x += sigmaChosen.offset
            else:
                y += sigmaChosen.offset

            self.listX.append(x)
            self.listY.append(y)

            print("x : " + str(x) + ",   y : " + str(y) + ",   direction : " + str(sigmaChosen))

            listSigma.append(sigmaChosen)


    def draw(self: 'ElephantWalk') -> None:
        if(self.lattice.dimension == 2):
            plt.plot(self.listX, self.listY)
        elif(self.lattice.dimension == 1):
            plt.plot(range(len(self.listX)), self.listX)
        plt.grid()
        plt.show()


    def calculateVariance(self: 'ElephantWalk') -> Any:
        if(self.lattice.dimension == 1):
            return self._calculateListVariance(self.listX)
        if(self.lattice.dimension == 2):
            listVarX: List[float] = []
            listVarY: List[float] = []
            listCoVar: List[float] = []
            for i in range(1, len(self.listX)):
                data = np.array([self.listX[:i],self.listY[:i]])
                covMatrix = np.cov(data,bias=True)
                listVarX.append(covMatrix[0][0])
                listVarY.append(covMatrix[1][1])
                listCoVar.append(covMatrix[0][1])
            return (listVarX, listVarY, listCoVar)


    def calculateTheoricalVariance(self: 'ElephantWalk') -> Optional[List[float]]:
        if(self.lattice.dimension == 1):
            variances: List[float] = []

            for i in range(1, len(self.listX) + 1):
                alpha = 2 * self.p - 1
                beta = 2 * self.q - 1
                moyenneQuadratique = i/(2 * alpha - 1) * (math.gamma(i + 2*alpha)/(math.gamma(i+1) * math.gamma(2*alpha))- 1)
                moyenne = beta  * (math.gamma(i + alpha) / (math.gamma(alpha + 1) * math.gamma(i)))
                # variances.append(moyenneQuadratique + moyenne*moyenne - 2*moyenne*moyenneQuadratique)
                variances.append(moyenneQuadratique - moyenne*moyenne)
            return variances
        return None





class RandomWalk(Walk):

    def __init__(self: 'RandomWalk', p:float, r:float, lattice:Lattice, seed: Any):
        """
        :param float p : la probabilité d'aller vers les positifs
        :param float r : la probabilité d'aller en vertical
        """
        super().__init__(lattice, seed)
        self.p = p
        self.r = r

        self.listX = []
        self.listY = []


    def walk(self: 'RandomWalk', stepNumber: int) -> None:

        x = 0
        y = 0

        self.listX = [x]
        self.listY = [y]

        for i in range(stepNumber):
            direc = Direction.NORTH
            if (not probab(self.r, self.rand)):
                direc = direc.changeDirection()
            if (not probab(self.p, self.rand)):
                direc = direc.reverse()

            listeAvailableDirections = self.lattice.getAvailableDirections(x, y)

            if(not direc in listeAvailableDirections):
                if(direc.reverse() in listeAvailableDirections):
                    direc = direc.reverse()
                elif(direc.changeDirection() in listeAvailableDirections):
                    direc = direc.changeDirection()
                else:
                    direc = listeAvailableDirections[0]

            # if (not direc in listeAvailableDirections):
            #     d = direc
            #     for direcF in listeAvailableDirections:
            #         if direc.isSameDirection(direcF):
            #             direc = direcF
            #     if (direc is d):
            #         for direcF in listeAvailableDirections:
            #             if direc.isSameSide(direcF):
            #                 direc = direcF
            #     if (direc is d):
            #         direc = listeAvailableDirections[0]

            
            if(direc.isHorizontal):
                x += direc.offset
            if(direc.isVertical):
                y += direc.offset

            self.listX.append(x)
            self.listY.append(y)


    def draw(self:'RandomWalk') -> None:
        if(self.lattice.dimension == 1):
            plt.plot(range(len(self.listX)), self.listX)
        if(self.lattice.dimension == 2):
            plt.plot(self.listX, self.listY)

        plt.grid()
        plt.show()


    def calculateVariance(self:'RandomWalk') -> Any:
        if(self.lattice.dimension == 1):
            return self._calculateListVariance(self.listX)
        return []

    def calculateTheoricalVariance(self:'RandomWalk') -> Optional[List[float]]:
        if(self.lattice.dimension == 1):
            variances:List[float] = []

            for i in range(len(self.listX)):
                variances.append(4 * i * self.q * (1 - self.q))

            return variances
        return None




class RandomWalkWithRandomJumps(Walk):
    def __init__(self: 'RandomWalkWithRandomJumps', p: float, q: float, r: float, s: float, lattice: Lattice, seed: Any, pi: Callable[[int, int], float] = lambda x,y : 0.5):
        super().__init__(lattice, seed)
        self.p = p
        self.q = q
        self.r = r
        self.s = s

        self.listX = []
        self.listY = []

    def walk(self: 'RandomWalkWithRandomJumps', stepNumber: int) -> None:

        x = 0
        y = 0

        self.listX = [x]
        self.listY = [y]

        for i in range(stepNumber):
            direc = Direction.NORTH
            if (probab(self.s, self.rand)):
                direc = direc.changeDirection()
            if (probab(self.q, self.rand)):
                direc = direc.reverse()

            listeAvailableDirections = self.lattice.getAvailableDirections(x, y)

            if (not direc in listeAvailableDirections):
                d = direc
                for direcF in listeAvailableDirections:
                    if direc.isSameDirection(direcF):
                        d = direcF
                if (direc is d):
                    direc = listeAvailableDirections[0]
                else:
                    direc = d

            if (direc.isHorizontal):
                x += direc.offset
            if (direc.isVertical):
                y += direc.offset

            self.listX.append(x)
            self.listY.append(y)

    def draw(self: 'RandomWalkWithRandomJumps') -> None:
        plt.plot(self.listX, self.listY)
        plt.grid()
        plt.show()


class RandomAlternativeWalk(Walk):
    def __init__(self: 'RandomAlternativeWalk', p: float, r: float, lattice: Lattice, seed: Any, probabilityFunc : Callable[[int, int], float] = lambda x, n : 1 if n == 0 else 1/n):
        """
        :param float p : la probabilité d'aller vers les positifs
        :param float r : la probabilité d'aller en vertical
        """
        super().__init__(lattice, seed)
        self.p = p
        self.r = r

        self.listX = []
        self.listY = []

        self.probabilityFunc = probabilityFunc

    def walk(self: 'RandomAlternativeWalk', stepNumber: int) -> None:

        x = 0
        y = 0

        self.listX = [x]
        self.listY = [y]

        for i in range(stepNumber):
            direc = Direction.NORTH
            if (not probab(self.r, self.rand)):
                direc = direc.changeDirection()
            if (not probab(self.p, self.rand)):
                direc = direc.reverse()

            listeAvailableDirections = self.lattice.getAvailableDirections(x, y)
            if(len(listeAvailableDirections) == 0):
                break

            if(not direc in listeAvailableDirections):
                if(direc.reverse() in listeAvailableDirections):
                    direc = direc.reverse()
                elif(direc.changeDirection() in listeAvailableDirections):
                    direc = direc.changeDirection()
                else:
                    direc = listeAvailableDirections[0]

            # if (not direc in listeAvailableDirections):
            #     d = direc
            #     for direcF in listeAvailableDirections:
            #         if direc.isSameDirection(direcF):
            #             d = direcF
            #     if (direc is d):
            #         direc = listeAvailableDirections[0]
            #     else:
            #         direc = d


            distanceToWalk = 1
            rng = self.rand.random()
            rngTotal = 0
            for j in range(1, i):
                nextRngTotal = rngTotal + self.probabilityFunc(j, i)
                if(rngTotal <= rng < nextRngTotal):
                    distanceToWalk = j
                    break
                rngTotal = nextRngTotal


            if (direc.isHorizontal):
                if(not self.lattice.isInfinite):
                    distanceToWalk = min(distanceToWalk, self.lattice.size / 2 - direc.offset * x)
                x += direc.offset * distanceToWalk
            if (direc.isVertical):
                if (not self.lattice.isInfinite):
                    distanceToWalk = min(distanceToWalk, self.lattice.size / 2 - direc.offset * y)
                y += direc.offset * distanceToWalk

            self.listX.append(x)
            self.listY.append(y)


    def draw(self: 'RandomAlternativeWalk') -> None:
        if(self.lattice.dimension == 1):
            plt.plot(range(len(self.listX)), self.listX)
        if(self.lattice.dimension == 2):
            plt.plot(self.listX, self.listY)

        plt.grid()
        plt.show()


def sudEast(d: 'Direction', x: int, rng: rand.Random) -> 'Direction':
    if(d.isVertical):
        return Direction.SOUTH
    if(d.isHorizontal):
        return Direction.EAST

def nordSudOuest(d: 'Direction', x: int, rng: rand.Random) -> 'Direction':
    if(d.isVertical):
        return Direction.NORTH if -100<x<100 else Direction.SOUTH
    if(d.isHorizontal):
        return Direction.WEST

#elephant = ElephantWalk1D(0, 0.5)

#RandomWalk :
#seedChosen = 836852345

def elephantWalk():
    seedChosen = rand.randrange(sys.maxsize)
    #seedChosen = 1259557030
    #seedChosen = 1816265777
    #seedChosen = 43316513
    elephant = ElephantWalk(0.8, 0.5, 0.5, 0.5, UnOrderedLattice(500, seedChosen), seedChosen)
    elephant.walk(100)
    print("Seed : " + str(seedChosen))
    elephant.draw()
    vPratique = elephant.calculateVariance()
    vTheorique = elephant.calculateTheoricalVariance()
    print(vPratique)
    print(vTheorique)
    plt.plot(range(len(vPratique)), vPratique, label="pas théorique")
    plt.plot(range(len(vPratique)), vTheorique, label="théorique")
    plt.grid()
    plt.legend()
    plt.show()



def elephantComparaison(probability:float = 0.5):
    seedChosen = rand.randrange(sys.maxsize)
    seedChosen = 204257849
    elephantNonOriente = ElephantWalk(probability, 0.5, probability, 0.5, UnOrderedLattice(2000, seedChosen), seedChosen)
    elephantOriente = ElephantWalk(probability, 0.5, probability, 0.5, OrderedLattice(2000, seedChosen), seedChosen)
    elephantNonOriente.walk(1000)
    elephantOriente.walk(1000)

    elephantNonOriente.draw()
    elephantOriente.draw()

    varXNonOriente, varYNonOriente, coVarNonOriente = elephantNonOriente.calculateVariance()
    varXOriente, varYOriente, coVarOriente = elephantOriente.calculateVariance()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.plot(range(len(varXNonOriente)), varXNonOriente, label="Non Orienté")
    ax1.plot(range(len(varXNonOriente)), varXOriente, label="Orienté")
    ax1.set_title("Variance en X")
    ax1.grid()
    ax1.legend()

    ax2.plot(range(len(varYNonOriente)), varYNonOriente, label="Non Orienté")
    ax2.plot(range(len(varYNonOriente)), varYOriente, label="Orienté")
    ax2.set_title("Variance en Y")
    ax2.grid()
    ax2.legend()

    ax3.plot(range(len(coVarNonOriente)), coVarNonOriente, label="Non Orienté")
    ax3.plot(range(len(coVarNonOriente)), coVarOriente, label="Orienté")
    ax3.set_title("Covariance")
    ax3.grid()
    ax3.legend()

    plt.show()

    print(seedChosen)

# seedChosen = rand.randrange(sys.maxsize)
# randomWalk = RandomWalk(1, 0.5, 0.5, 0.5, UnOrderedLattice1D(500, seedChosen), seedChosen)
# randomWalk.walk(100)
# randomWalk.draw()
# vPratique = randomWalk.calculateVariance()
# vTheorique = randomWalk.calculateTheoricalVariance()
# print(vPratique)
# print(vTheorique)
# plt.plot(range(len(vPratique)), vPratique, label="pas théorique")
# plt.plot(range(len(vTheorique)), vTheorique, label="théorique")
# plt.grid()
# plt.legend()
# plt.show()

elephantComparaison()
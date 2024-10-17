# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from fitFunction import aucCal
# D: Dimension, P: Number of initial population, G: Number of iterations, ub: Upper bound of variable values, lb: Lower bound of variable values
class PO():
    def __init__(self, fitness, D=30, P=20, G=500, ub=1, lb=0):
        self.fitness = fitness
        self.D = D
        self.P = P
        self.G = G
        self.ub = ub * np.ones([self.P, self.D])
        self.lb = lb * np.ones([self.P, self.D])

        self.gbest_X = np.zeros([self.D])
        self.gbest_F = np.inf
        self.loss_curve = np.zeros(self.G)

    def opt(self):
        # Initialization
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=[self.P, self.D])
        F = np.zeros(self.P)   # Store various sub-fitness functions
        # self.X = np.random.rand(self.P, self.D) * (self.ub - self.lb) + self.lb
        # Iteration
        for g in range(self.G):
            # Fitness value calculation
            for p in range(self.P):
                F[p] = self.fitness(self.X[p,:])
            # Update the best solution
            if np.min(F) < self.gbest_F:
                idx = F.argmin()              # Store the index value
                self.gbest_X = self.X[idx].copy()     # Store the position of the optimal solution
                self.gbest_F = F.min()                # Store the optimal solution

            # print('Iteration {}: Optimal value is:{}'.format(g, 1/(self.gbest_F+1e-10)))
            # Calculate the optimal value based on the best solution
            # print('Iteration {}: Optimal solution value is:{}'.format(g, self.gbest_X))
            f = self.fitness(self.gbest_X)
            # print('Iteration {}: Optimal value calculated based on the best solution is:{}'.format(g, 1/(f+1e-10)))

            # print('Iteration {}: Error optimal value is:{}'.format(g, self.gbest_F))
            # auc = aucCal(self.gbest_X)
            # print('Iteration {}: AUC optimal value is:{}'.format(g, auc))
            # Convergence curve
            self.loss_curve[g] = self.gbest_F     # Store the optimal solution curve for each iteration

            for i in range(self.P):
                St = np.random.randint(1, 5)
                # Foraging behavior
                if St == 1:
                    self.X[i, :] = (self.X[i, :] - self.gbest_X) * self.Levy() + np.random.rand() * np.mean(self.X[i,:]) * (
                            1 - g / self.G) ** (2 * g / self.G)
                # Staying behavior
                elif St == 2:
                    self.X[i,:] = self.X[i,:] + self.gbest_X * self.Levy() + np.random.randn() * (1 - g / self.G) * np.ones(self.D)
                # Communication behavior
                elif St == 3:
                    H = np.random.rand()
                    if H < 0.5:
                        self.X[i, :] = self.X[i,:] + np.random.rand() / 5 * (1 - g / self.G) * (self.X[i,:] - np.mean(self.X[i,:]))
                    else:
                        self.X[i,:] = self.X[i,:] + np.random.rand() / 5 * (1 - g / self.G) * np.exp(-g / (np.random.rand() * self.G))
                else:
                    self.X[i,:] = self.X[i,:] + np.random.rand() * np.cos((np.pi * g) / (2 * self.G)) * (
                            self.gbest_X - self.X[i,:]) - np.cos(np.random.rand() * np.pi) * (g / self.G) ** (2 / self.G) * (
                                          self.X[i, :] - self.gbest_X)
                # Select the best
                # for p in range(self.P):
                #     F[p] = self.fitness(self.X[p, :])
                # index = np.argsort(F)
                # # self.X = self.X[index,:]
                # for j in range(self.P):
                #     self.X[j,:] = self.X[index[j],:]

            # Boundary handling
            self.X = np.clip(self.X, self.lb, self.ub)

    def Levy(self):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
                np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(1, self.D) * sigma
        v = np.random.randn(1, self.D)
        step = u / np.power(np.abs(v), (1 / beta))
        return step

    def plot_curve(self):
        plt.figure()
        plt.title('Loss Curve')
        plt.plot(self.loss_curve, label='Loss')
        plt.grid()
        plt.legend()
        plt.show()

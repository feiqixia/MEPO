# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


class WOA():
    def __init__(self, fitness, D=30, P=20, G=500, ub=1, lb=0,
                 b=1, a_max=2, a_min=0, a2_max=-1, a2_min=-2, l_max=1, l_min=-1):
        self.fitness = fitness
        self.D = D
        self.P = P
        self.G = G
        self.ub = ub * np.ones([self.P, self.D])
        self.lb = lb * np.ones([self.P, self.D])
        self.a_max = a_max
        self.a_min = a_min
        self.a2_max = a2_max
        self.a2_min = a2_min
        self.l_max = l_max
        self.l_min = l_min
        self.b = b

        self.gbest_X = np.zeros([self.D])
        self.gbest_F = np.inf
        self.loss_curve = np.zeros(self.G)

    def opt(self):
        # Initialization
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=[self.P, self.D])
        F = np.zeros(self.P)  # Store various sub-fitness functions
        # Iteration
        for g in range(self.G):
            # Fitness calculation
            # F = self.fitness(self.X)
            for p in range(self.P):
                F[p] = self.fitness(self.X[p, :])
            # Update the best solution
            if np.min(F) < self.gbest_F:
                idx = F.argmin()
                self.gbest_X = self.X[idx].copy()
                self.gbest_F = F.min()

            # Convergence curve
            self.loss_curve[g] = self.gbest_F

            # Update
            a = self.a_max - (self.a_max - self.a_min) * (g / self.G)
            a2 = self.a2_max - (self.a2_max - self.a2_min) * (g / self.G)

            for i in range(self.P):
                p = np.random.uniform()
                r1 = np.random.uniform()
                r2 = np.random.uniform()
                r3 = np.random.uniform()
                A = 2 * a * r1 - a  # (2.3)
                C = 2 * r2  # (2.4)
                l = (a2 - 1) * r3 + 1  # (???)

                if p > 0.5:
                    D = np.abs(self.gbest_X - self.X[i, :])
                    self.X[i, :] = D * np.exp(self.b * l) * np.cos(2 * np.pi * l) + self.gbest_X  # (2.5)
                else:
                    if np.abs(A) < 1:
                        D = np.abs(C * self.gbest_X - self.X[i, :])  # (2.1)
                        self.X[i, :] = self.gbest_X - A * D  # (2.2)
                    else:
                        idx = np.random.randint(low=0, high=self.P)
                        X_rand = self.X[idx]
                        D = np.abs(C * X_rand - self.X[i, :])  # (2.7)
                        self.X[i, :] = X_rand - A * D  # (2.8)

            # Boundary handling
            self.X = np.clip(self.X, self.lb, self.ub)

    def plot_curve(self):
        plt.figure()
        plt.title('loss curve [' + str(round(self.gbest_F, 3)) + ']')
        plt.plot(self.loss_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()

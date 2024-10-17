# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


class CPO():
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
        F = np.zeros(self.P)  # Store various sub-fitness functions
        # Iteration
        for g in range(self.G):
            # Fitness calculation
            for p in range(self.P):
                F[p] = self.fitness(self.X[p,:])

            # Update the best solution
            if np.min(F) < self.gbest_F:
                idx = F.argmin()
                self.gbest_X = self.X[idx].copy()
                self.gbest_F = F.min()

            # Convergence curve
            self.loss_curve[g] = self.gbest_F

            for i in range(self.P):
                total_sum = 0
                U1 = np.random.rand(self.D) > np.random.rand()
                # Exploration phase, the following are the first and second defense strategies
                if np.random.rand() < np.random.rand():
                    # First defense strategy
                    if np.random.rand() < np.random.rand():
                        y = (self.X[i,:] + self.X[np.random.randint(self.P),:]) / 2
                        self.X[i,:] = self.X[i,:] + np.random.randn(self.D) * np.abs(2 * np.random.rand() * self.gbest_X - y)
                    # Second defense strategy
                    else:
                        y = (self.X[i, :] + self.X[np.random.randint(self.P), :]) / 2
                        self.X[i,:] = U1 * self.X[i,:] + (1 - U1) * (y + np.random.rand() * (self.X[np.random.randint(self.P)] - self.X[np.random.randint(self.P)]))

                # Development phase, the following are the 3rd and 4th defense stages
                else:
                    # 80% in the 3rd defense stage
                    Yt = 2 * np.random.rand() * (1 - g / self.G) ** (g / self.G)
                    U2 = np.random.rand(self.D) < 0.5 * 2 - 1
                    S = np.random.rand() * U2
                    if np.random.rand() < 0.8:
                        # Calculate the total fitness
                        for p in range(self.P):
                            total_sum += self.fitness(self.X[p, :])
                        St = np.exp(self.fitness(self.X[i,:]) / (total_sum + np.finfo(float).eps))
                        S = S * Yt * St
                        self.X[i,:] = (1 - U1) * self.X[i,:] + U1 * (self.X[np.random.randint(self.P), i] + St * (
                                self.X[np.random.randint(self.P), i] - self.X[np.random.randint(self.P), i]) - S)
                    # 4th defense stage
                    else:
                        # Calculate the total fitness
                        for p in range(self.P):
                            total_sum += self.fitness(self.X[p, :])
                        Mt = np.exp(self.fitness(self.X[i,:]) / (total_sum + np.finfo(float).eps))
                        vt = self.X[i,:]
                        Vtp = self.X[np.random.randint(self.P),:]
                        Ft = np.random.rand(self.D) * (Mt * (-vt + Vtp))
                        S = S * Yt * Ft
                        self.X[i,:] = (self.gbest_X + (0.2 * (1 - np.random.rand()) + np.random.rand()) * (U2 * self.gbest_X - self.X[i,:])) - S
                N_min = 120
                T = 2
                # self.P = int(N_min + (self.P - N_min) * (1 - (g % (self.G / T)) / (self.G / T)))

            # Boundary handling
            self.X = np.clip(self.X, self.lb, self.ub)

    def plot_curve(self):
        plt.figure()
        plt.title('loss curve [' + str(round(self.gbest_F, 3)) + ']')
        plt.plot(self.loss_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()

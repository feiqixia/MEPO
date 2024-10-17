# Elite Opposition-Based Parrot Optimization Algorithm
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# D: Dimensions, P: Number of initialized populations, G: Number of iterations, ub: Upper bound of variable values, lb: Lower bound of variable values
class EPO():
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
        # self.X = np.random.uniform(low=self.lb, high=self.ub, size=[self.P, self.D])
        # Using tent chaotic mapping for initialization
        self.X = initialize_populations(num_populations=self.P, population_size=self.D, lb=self.lb, ub=self.ub)

        # Iteration
        for g in range(self.G):
            # OBL
            self.X, F = self.OBL()
            # Calculate fitness values
            for p in range(self.P):
                F[p] = self.fitness(self.X[p, :])
            # Update the best solution
            if np.min(F) < self.gbest_F:
                idx = F.argmin()  # Store the index
                self.gbest_X = self.X[idx].copy()  # Store the position of the best solution
                self.gbest_F = F.min()  # Store the best solution value
            # print('Iteration {}: Best value is:{}'.format(g, 1 / (self.gbest_F + 1e-10)))
            # print('Iteration {}: Best value is:{}'.format(g, self.gbest_F))
            # Convergence curve
            self.loss_curve[g] = self.gbest_F  # Store the best solution curve for each iteration

            for i in range(self.P):
                St = np.random.randint(1, 5)
                # Foraging behavior
                if St == 1:
                    self.X[i, :] = (self.X[i, :] - self.gbest_X) * self.Levy() + np.random.rand() * np.mean(
                        self.X[i, :]) * (
                                           1 - g / self.G) ** (2 * g / self.G)
                # Staying behavior
                elif St == 2:
                    self.X[i, :] = self.X[i, :] + self.gbest_X * self.Levy() + np.random.randn() * (
                                1 - g / self.G) * np.ones(self.D)
                # Communicating behavior
                elif St == 3:
                    H = np.random.rand()
                    if H < 0.5:
                        self.X[i, :] = self.X[i, :] + np.random.rand() / 5 * (1 - g / self.G) * (
                                    self.X[i, :] - np.mean(self.X[i, :]))
                    else:
                        self.X[i, :] = self.X[i, :] + np.random.rand() / 5 * (1 - g / self.G) * np.exp(
                            -g / (np.random.rand() * self.G))
                else:
                    self.X[i, :] = self.X[i, :] + np.random.rand() * np.cos((np.pi * g) / (2 * self.G)) * (
                            self.gbest_X - self.X[i, :]) - np.cos(np.random.rand() * np.pi) * (g / self.G) ** (
                                               2 / self.G) * (
                                           self.X[i, :] - self.gbest_X)
                # Perform t-distribution mutation
                X_New = self.X[i, :] + stats.t.rvs(g + 1, size=self.D) * self.X[i, :]
                if (self.fitness(X_New) < self.fitness(self.X[i, :])):
                    self.X[i, :] = X_New

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
        plt.title('loss curve')
        plt.plot(self.loss_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()

    def OBL(self):
        # Generate opposite solutions
        k = np.random.uniform()
        alpha = self.X.min(axis=0)
        beta = self.X.max(axis=0)
        obl_X = k * (alpha + beta) - self.X  # (5)

        # Boundary handling for opposite solutions
        rand_X = np.random.uniform(low=alpha, high=beta, size=[self.P, self.D])  # (6)
        mask = np.logical_or(obl_X > self.ub, obl_X < self.lb)
        obl_X[mask] = rand_X[mask].copy()

        # Get new solutions
        concat_X = np.vstack([obl_X, self.X])
        # Calculate fitness values
        F = np.zeros(2 * self.P)
        for p in range(self.P * 2):
            F[p] = self.fitness(concat_X[p, :])
        top_idx = F.argsort()[:self.P]
        top_F = F[top_idx].copy()
        top_X = concat_X[top_idx].copy()

        return top_X, top_F


# Tent chaotic mapping function
def tent_map(x, mu=2):
    if x < 0.5:
        return mu * x
    else:
        return mu * (1 - x)


# Generate a tent chaotic sequence of a specified length
def generate_tent_chaos_sequence(length, mu=2, initial_value=0.5):
    sequence = np.zeros(length)
    x = initial_value
    for i in range(length):
        x = tent_map(x, mu)
        sequence[i] = x
    return sequence


def initialize_populations(num_populations, population_size, lb, ub, mu=2, initial_value=0.5):
    """
    Initialize multiple populations using Tent chaotic mapping
    """
    # Generate a long enough chaotic sequence
    chaos_sequence = generate_tent_chaos_sequence(num_populations * population_size, mu, initial_value)

    # Rearrange the chaotic sequence into multiple populations
    populations = np.reshape(chaos_sequence, (num_populations, population_size))

    # Map the values of the chaotic sequence to the desired range, e.g., [-1, 1]
    # Here, assuming we want to map values to the range [-1, 1]
    populations = (ub - lb) * populations + lb
    return populations

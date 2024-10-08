# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:59:58 2020

@author: ZongSing_NB

Main reference:
https://seyedalimirjalili.com/woa
https://doi.org/10.1016/j.advengsoft.2016.01.008
"""

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
        # 初始化
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=[self.P, self.D])
        F = np.zeros(self.P)  # 存放各种子适应度函数
        # 迭代
        for g in range(self.G):
            # 適應值計算
            for p in range(self.P):
                F[p] = self.fitness(self.X[p,:])

            # 更新最佳解
            if np.min(F) < self.gbest_F:
                idx = F.argmin()
                self.gbest_X = self.X[idx].copy()
                self.gbest_F = F.min()

            # 收斂曲線
            self.loss_curve[g] = self.gbest_F

            for i in range(self.P):
                sum = 0
                U1 = np.random.rand(self.D) > np.random.rand()
                #勘探阶段，如下为第一及第二防御策略
                if np.random.rand() < np.random.rand():
                    #第一防御阶段
                    if np.random.rand() < np.random.rand():
                        y = (self.X[i,:] + self.X[np.random.randint(self.P),:]) / 2
                        self.X[i,:] = self.X[i,:] + np.random.randn(self.D) * np.abs(2 * np.random.rand() * self.gbest_X - y)
                    #第二防御策略
                    else:
                        y = (self.X[i, :] + self.X[np.random.randint(self.P), :]) / 2
                        self.X[i,:] = U1 * self.X[i,:] + (1 - U1) * (y + np.random.rand() * (self.X[np.random.randint(self.P)] - self.X[np.random.randint(self.P)]))

                #开发阶段，如下为第3及第4防御阶段
                else:
                    #80%在第3防御阶段
                    Yt = 2 * np.random.rand() * (1 - g / self.G) ** (g / self.G)
                    U2 = np.random.rand(self.D) < 0.5 * 2 - 1
                    S = np.random.rand() * U2
                    if np.random.rand() < 0.8:
                        #计算适应度总和
                        for p in range(self.P):
                            sum = sum + self.fitness(self.X[p, :])
                        St = np.exp(self.fitness(self.X[i,:]) / (sum + np.finfo(float).eps))
                        S = S * Yt * St
                        self.X[i,:] = (1 - U1) * self.X[i,:] + U1 * (self.X[np.random.randint(self.P),i] + St * (
                                self.X[np.random.randint(self.P), i] - self.X[np.random.randint(self.P),i]) - S)
                    #第4防御阶段
                    else:
                        #计算适应度总和
                        for p in range(self.P):
                            sum = sum + self.fitness(self.X[p, :])
                        Mt = np.exp(self.fitness(self.X[i,:]) / (sum + np.finfo(float).eps))
                        vt = self.X[i,:]
                        Vtp = self.X[np.random.randint(self.P),:]
                        Ft = np.random.rand(self.D) * (Mt * (-vt + Vtp))
                        S = S * Yt * Ft
                        self.X[i,:] = (self.gbest_X + (0.2 * (1 - np.random.rand()) + np.random.rand()) * (U2 * self.gbest_X - self.X[i,:])) - S
                N_min = 120
                T = 2
                # self.P = int(N_min + (self.P - N_min) * (1 - (g % (self.G / T)) / (self.G / T)))

            # 邊界處理
            self.X = np.clip(self.X, self.lb, self.ub)

    def plot_curve(self):
        plt.figure()
        plt.title('loss curve [' + str(round(self.gBest_curve[-1], 3)) + ']')
        plt.plot(self.gBest_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()

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
from fitFunction import aucCal
#D:维度 P:初始化种群个数 G:迭代次数 ub:变量取值上界  lb：变量取值下界
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
        # 初始化
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=[self.P, self.D])
        F = np.zeros(self.P)   #存放各种子适应度函数
        # self.X = np.random.rand(self.P, self.D) * (self.ub - self.lb) + self.lb
        # 迭代
        for g in range(self.G):
            # 適應值計算
            for p in range(self.P):
                F[p] = self.fitness(self.X[p,:])
            # 更新最佳解
            if np.min(F) < self.gbest_F:
                idx = F.argmin()              #存储索引值
                self.gbest_X = self.X[idx].copy()     #存储最优解的位置
                self.gbest_F = F.min()                #存储最优解

            # print('第{}次迭代，最优值为:{}'.format(g, 1/(self.gbest_F+1e-10)))
            #依据最优解求取最优值
            # print('第{}次迭代，最优解值为:{}'.format(g, self.gbest_X))
            f = self.fitness(self.gbest_X)
            # print('第{}次迭代，依据最优解求取的最优值为:{}'.format(g, 1/(f+1e-10)))

            # print('第{}次迭代，err最优值为:{}'.format(g, self.gbest_F))
            # auc = aucCal(self.gbest_X)
            # print('第{}次迭代，auc最优值为:{}'.format(g, auc))
            # 收斂曲線
            self.loss_curve[g] = self.gbest_F     #存储各迭代次数对应的最优解曲线

            for i in range(self.P):
                St = np.random.randint(1, 5)
                #觅食行为
                if St == 1:
                    self.X[i, :] = (self.X[i, :] - self.gbest_X) * self.Levy() + np.random.rand() * np.mean(self.X[i,:]) * (
                            1 - g / self.G) ** (2 * g / self.G)
                #停留行为
                elif St == 2:
                    self.X[i,:] = self.X[i,:] + self.gbest_X * self.Levy() + np.random.randn() * (1 - g / self.G) * np.ones(self.D)
                #交流行为
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
                #选择最优
                # for p in range(self.P):
                #     F[p] = self.fitness(self.X[p, :])
                # index = np.argsort(F)
                # # self.X = self.X[index,:]
                # for j in range(self.P):
                #     self.X[j,:] = self.X[index[j],:]

            # 邊界處理
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



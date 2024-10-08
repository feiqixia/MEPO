#精英反向鹦鹉优化算法
# -*- coding: utf-8 -*-
'''
1、使用精英反向效果不好
2、头部使用tent混沌映射
'''
"""
Created on Thu Jul 16 21:59:58 2020

@author: ZongSing_NB

Main reference:
https://seyedalimirjalili.com/woa
https://doi.org/10.1016/j.advengsoft.2016.01.008
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
#D:维度 P:初始化种群个数 G:迭代次数 ub:变量取值上界  lb：变量取值下界
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
        # 初始化
        # self.X = np.random.uniform(low=self.lb, high=self.ub, size=[self.P, self.D])
        #使用tent混沌映射进行初始化
        self.X = initialize_populations(num_populations = self.P, population_size = self.D, lb=self.lb, ub=self.ub)
        # 迭代
        for g in range(self.G):
            # OBL
            self.X, F = self.OBL()
            # 適應值計算
            for p in range(self.P):
                F[p] = self.fitness(self.X[p,:])
            # 更新最佳解
            if np.min(F) < self.gbest_F:
                idx = F.argmin()              #存储索引值
                self.gbest_X = self.X[idx].copy()     #存储最优解的位置
                self.gbest_F = F.min()                #存储最优解
            # print('第{}次迭代，最优值为:{}'.format(g, 1 / (self.gbest_F + 1e-10)))
            # print('第{}次迭代，最优值为:{}'.format(g, self.gbest_F))
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
                #进行t分布变异
                X_New = self.X[i,:] + stats.t.rvs(g+1,size=self.D) * self.X[i,:]
                if(self.fitness(X_New) < self.fitness(self.X[i,:])):
                    self.X[i,:] = X_New
                #选择最优
                # F = self.fitness(self.X)
                # index = np.argsort(F)
                # self.X = self.X[index,:]
                #此段写法挺有意思，必须如此写，才有效果
                #index = [4,2,6,1,3,5,0]   0:4,1:2,3:6,4:2,5:5,6:4
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

    def OBL(self):
        # 產生反向解
        k = np.random.uniform()
        alpha = self.X.min(axis=0)
        beta = self.X.max(axis=0)
        obl_X = k * (alpha + beta) - self.X  # (5)

        # 對反向解進行邊界處理
        rand_X = np.random.uniform(low=alpha, high=beta, size=[self.P, self.D])  # (6)
        mask = np.logical_or(obl_X > self.ub, obl_X < self.lb)
        obl_X[mask] = rand_X[mask].copy()

        # 取得新解
        concat_X = np.vstack([obl_X, self.X])
        # 適應值計算
        F = np.zeros(2 * self.P)
        for p in range(self.P * 2):
            F[p] = self.fitness(concat_X[p,:])
        top_idx = F.argsort()[:self.P]
        top_F = F[top_idx].copy()
        top_X = concat_X[top_idx].copy()
        # top_X = concat_X[top_idx].copy()

        return top_X, top_F

#tent混沌映射函数
def tent_map(x, mu = 2):
    if x < 0.5:
        return mu * x
    else:
        return mu * (1 - x)

#生成指定长度的Tent混沌序列
def generate_tent_chaos_sequence(length, mu=2, initial_value=0.5):
    sequence = np.zeros(length)
    x = initial_value
    for i in range(length):
        x = tent_map(x, mu)
        sequence[i] = x
    return sequence


def initialize_populations(num_populations, population_size, lb, ub, mu=2, initial_value=0.5):
    """
    利用Tent混沌映射初始化多个种群
    """
    # 生成一个足够长的混沌序列
    chaos_sequence = generate_tent_chaos_sequence(num_populations * population_size, mu, initial_value)

    # 将混沌序列重新排列为多个种群
    populations = np.reshape(chaos_sequence, (num_populations, population_size))

    # 将混沌序列的值映射到所需的范围，例如[-1, 1]
    # 这里假设我们想要将值映射到[-1, 1]范围
    populations = (ub-lb) * populations + lb
    return populations
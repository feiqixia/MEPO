import time
import numpy as np
import pandas as pd
from PO import PO    #基线算法1
from EPO import EPO
from WOA import WOA  #基线算法2
from CPO import CPO  #基线算法3
from fitFunction import Fun
from fitFunction import aucCal

D = 20
G = 50
P = 10
run_times = 30
table = pd.DataFrame(np.zeros(4), index=['avg', 'std',  'best', 'time'])
loss_curves = np.zeros(G)
F_table = np.zeros(run_times)
solve_table = []   #存储最优解
AUC_table = np.zeros(run_times)

for t in range(run_times):
    ub = 1 * np.ones(D)
    lb = 0 * np.ones(D)
    optimizer =WOA(fitness=Fun,D=D, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t] = 1/(optimizer.gbest_F + np.finfo(float).eps)        #保留本次最优auc
    table[0]['avg'] += 1/(optimizer.gbest_F + np.finfo(float).eps)     #累加各次的最优auc
    table[0]['time'] += ed - st                                        #计算时间
    loss_curves += 1/(optimizer.loss_curve + np.finfo(float).eps)
    print('第{}次:最终auc value为:{}'.format(t,1/(optimizer.gbest_F + np.finfo(float).eps)))
    num_feature = np.round(optimizer.gbest_X, 0).astype(int)
    solve_table.append(num_feature)
    # print('最终选取的特征个数为:',np.sum(num_feature == 1))
    # 找到数组中为1的元素的索引
    indices_ones = np.where(num_feature == 1)
    # print('选取的索引为:',indices_ones)
    # print('auc value is:', aucCal(optimizer.gbest_X))

table.loc['best'] = F_table.max(axis=0)    #最优值
max_index = np.argmax(F_table)             #获取最优值的索引
solve_best = np.array(solve_table[int(max_index)])
indices_ones = np.where(solve_best == 1)
print('最终选取的索引为：',indices_ones)
# print('最终auc值为:',table.loc['best'])
table.loc[['avg', 'time']] = table.loc[['avg', 'time']] / run_times  #存储平均值及平均运行时间
table.loc['std'] = F_table.std(axis=0)   #存储方差
table.to_csv('table(EPO).csv')
print(table)

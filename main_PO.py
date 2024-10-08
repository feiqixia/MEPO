import time
import numpy as np
import pandas as pd
from PO import PO
from EPO import EPO
from fitFunction import Fun
from fitFunction import aucScore
D = 20
G = 100
P = 10
run_times = 1
table = pd.DataFrame(np.zeros([6, 36]), index=['avg', 'std', 'worst', 'best', 'ideal', 'time'])
loss_curves = np.zeros([G, 36])
F_table = np.zeros([run_times, 36])

for t in range(run_times):
    item = 0
    ub = 1 * np.ones(D)
    lb = 0 * np.ones(D)
    optimizer = PO(fitness=Fun,D=D, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = 1/(optimizer.gbest_F + np.finfo(float).eps)
    table[item]['avg'] += 1/(optimizer.gbest_F + np.finfo(float).eps)
    table[item]['time'] += ed - st
    loss_curves[:, item] += 1/(optimizer.loss_curve + np.finfo(float).eps)
    selectX = optimizer.gbest_X  #选取的最优解

    print('最终最优值为:',1/(optimizer.gbest_F + np.finfo(float).eps))
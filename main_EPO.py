import time
import numpy as np
import pandas as pd
from PO import PO    # Baseline algorithm 1
from EPO import EPO
from WOA import WOA  # Baseline algorithm 2
from CPO import CPO  # Baseline algorithm 3
from fitFunction import Fun
from fitFunction import aucCal

D = 20
G = 50
P = 10
run_times = 30
table = pd.DataFrame(np.zeros(4), index=['avg', 'std', 'best', 'time'])
loss_curves = np.zeros(G)
F_table = np.zeros(run_times)
solve_table = []   # Store the optimal solution
AUC_table = np.zeros(run_times)

for t in range(run_times):
    ub = 1 * np.ones(D)
    lb = 0 * np.ones(D)
    optimizer = EPO(fitness=Fun, D=D, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t] = 1/(optimizer.gbest_F + np.finfo(float).eps)  # Store the current optimal AUC
    table[0]['avg'] += 1/(optimizer.gbest_F + np.finfo(float).eps)  # Accumulate the optimal AUC for each run
    table[0]['time'] += ed - st  # Calculate the total time
    loss_curves += 1/(optimizer.loss_curve + np.finfo(float).eps)
    print('Run {}: Final AUC value: {}'.format(t, 1/(optimizer.gbest_F + np.finfo(float).eps)))
    num_feature = np.round(optimizer.gbest_X, 0).astype(int)
    solve_table.append(num_feature)

    indices_ones = np.where(num_feature == 1)

table.loc['best'] = F_table.max(axis=0)  # Store the best value
max_index = np.argmax(F_table)  # Get the index of the best value
solve_best = np.array(solve_table[int(max_index)])
indices_ones = np.where(solve_best == 1)
print('Final selected indices: ', indices_ones)
# print('Final AUC value:', table.loc['best'])
table.loc[['avg', 'time']] = table.loc[['avg', 'time']] / run_times  # Store the average values and average running time
table.loc['std'] = F_table.std(axis=0)  # Store the standard deviation
table.to_csv('table(EPO).csv')
print(table)

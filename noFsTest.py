import time
import numpy as np
import pandas as pd
from PO import PO    #基线算法1
from EPO import EPO
from WOA import WOA  #基线算法2
from CPO import CPO  #基线算法3
from fitFunction import Fun
from fitFunction import aucCal


x = np.zeros(20)
indices_to_set = [ 7, 11, 12, 17]
# 使用循环将指定索引位置的值设为1
for idx in indices_to_set:
    x[idx] = 1 
cost = Fun(x)

print('cost is:',1/(cost+1E-100))
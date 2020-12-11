from datetime import datetime
import argparse
import os

import numpy as np
import pandas as pd
from metrics import *

root_path = './GNN_results/'
dirs = [ d for d in os.listdir(root_path) if d.startswith('.')==False]
dirs.sort()
print(dirs)

res_data = {}

# load data
_i=0
for d in dirs:
  dir = root_path + d + '/'
  try:
    pred = np.loadtxt(dir + 'pred')
  except:
    pred = np.load(dir + 'pred')
  try:
    truth = np.loadtxt(dir + 'true')
  except:
    truth = np.load(dir + 'true')

  dimension = truth.shape[0]
  var_names = ['x{}'.format(i) for i in range(dimension)]

  #'''
  for i,j in np.ndindex(dimension,dimension):
    if pred[i][j] != 0:
      pred[i][j] = 1
    else:
      pred[i][j] = 0

    if truth[i][j] != 0:
      truth[i][j] = 1
    else:
      truth[i][j] = 0
  #'''

  print('#'*78)
  print(f'\t\tRESULTS FOR {d}')
  print('#'*78)
  performance = getAllMetrics(pred, truth)
  for k in performance.keys():
    print(k,':\t', performance[k])

  res_data[_i] = [ d, performance['Accuracy'], performance['F1 Score'], performance['Precision'], performance['Recall'] ]
  _i = _i + 1
  #print(res_data)

  #drawGraph(pred, var_names)
  #drawGraph(truth, var_names)

df = pd.DataFrame.from_dict(res_data, orient='index')
print(df.head())
df.to_csv('./gnn_res_data.csv',index=False)

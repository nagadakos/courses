from datetime import datetime
import argparse
import os

import numpy as np
import pandas as pd
from metrics import *

root_path = './lingam/output/'

dirs = [ d for d in os.listdir(root_path) if d.startswith('.')==False]
dirs.sort()
print(dirs)

res_data = {}

# load data
i=0
for d in dirs:
  dir = root_path + d + '/'
  pred = np.load(dir + 'prediction.npy')
  truth = np.load(dir + 'ground_truth.npy').T

  dimension = truth.shape[0]
  var_names = ['x{}'.format(i) for i in range(dimension)]

  #'''
  for x,y in np.ndindex(dimension,dimension):
    if pred[x][y] != 0:
      pred[x][y] = 1
    else:
      pred[x][y] = 0

    if truth[x][y] != 0:
      truth[x][y] = 1
    else:
      truth[x][y] = 0
  #'''

  print('#'*78)
  print(f'\t\tRESULTS FOR {d}')
  print('#'*78)
  performance = getAllMetrics(pred, truth)
  for k in performance.keys():
    print(k,':\t', performance[k])

  res_data[i] = [ d, performance['Accuracy'], performance['F1 Score'], performance['Precision'], performance['Recall'] ]
  i = i + 1

  #drawGraph(pred, var_names)
  #drawGraph(truth, var_names)

df = pd.DataFrame.from_dict(res_data, orient='index')
print(df)
df.to_csv('./lingam_res_data.csv',index=False)


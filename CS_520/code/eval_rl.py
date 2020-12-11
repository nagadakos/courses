from datetime import datetime
import argparse
import os

import numpy as np
import pandas as pd
from metrics import *

import matplotlib.pyplot as plt

root_path = './trustworthyAI/Causal_Structure_Learning/Causal_Discovery_RL/src/output/'

dirs = [ d for d in os.listdir(root_path) if d.startswith( ('.','_') )==False]
dirs.sort()
print(dirs)

res_data = {}

# load data
_i = 0
for d in dirs:
  dir = root_path + d + '/results/'
  result_files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
  result_files = [f for f in result_files if 'pruned' in f]
  result_files.sort(key=lambda x: int(x.split('_')[0]))

  truth = np.load(root_path + d + '/gt.npy')
  dimension = truth.shape[0]

  if 'cyto' in d:
    var_names = ['praf', 'pmek', 'plcg', 'pip2', 'pip3', 'p44/42', 'pakts473', 'PKA', 'PKC', 'P38', 'pjnk']
  else:
    var_names = ['x{}'.format(i) for i in range(dimension)]

  res = {}

  best = None
  best_pred = None

  for f in result_files:
    pred = np.load(dir + f)
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
    performance = getAllMetrics(pred, truth)
    performance['Iteration'] = int(f.split('_')[0])

    for k in performance.keys():
      if k not in res.keys():
        res[k] = []
      res[k].append(performance[k])

    if best == None or performance['F1 Score'] > best['F1 Score']:
      best = performance
      best_pred = pred

  print('#'*78)
  print(f'\t\tRESULTS FOR {d}')
  print('#'*78)
  for k in performance.keys():
    print('\t',k,':\t', performance[k])
    if k != 'Iteration':
      plt.plot(res['Iteration'], res[k], label=k)

  res_data[_i] = [ d, performance['Accuracy'], performance['F1 Score'], performance['Precision'], performance['Recall'] ]
  _i = _i + 1

  print(_i, len(res_data.keys()))

  print('BEST:')
  for k in performance.keys():
    print('\t',k,':\t', best[k])

  '''
  plt.xlabel('Iteration')
  plt.ylabel('Score')

  plt.ylim(-0.01, 1)
  title_str = d.replace('_',' ').replace('lingam', 'LiNGAM noise,').replace('gaussian', 'Gaussian noise,')
  title_str = title_str.replace('var', '').replace('same','same variance').replace('different','different variance')
  title_str = title_str.replace('cyto', 'Cytometric')
  plt.title(title_str)
  plt.legend()
  plt.show()
  '''

  #drawGraph(pred, var_names)
  #drawGraph(truth, var_names)

df = pd.DataFrame.from_dict(res_data, orient='index')
print(df)
print(df.shape)
df.to_csv('./rl_res_data.csv',index=False)

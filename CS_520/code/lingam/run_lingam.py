from datetime import datetime
import argparse
import os

import numpy as np
import pandas as pd
import graphviz
import lingam
from lingam.utils import print_causal_directions, print_dagc, make_dot
#from metrics import *

parser = argparse.ArgumentParser()
parser.add_argument('-d','--data', type=str, default='./')
parser.add_argument('-s','--save', type=str, default='./')
args = parser.parse_args()

#file_path = '../../data/sythetic/data_lingam_n5000_d12_s1_p0.5_seed8/'
#file_path = '../../data/sythetic/data_gaussian_n5000_d12_s1_p0.5_seed8/'
#file_path = '../../data/sythetic/data_lingam_n5000_d12_s0_p0.5_seed9/'
#file_path = '../../data/sythetic/data_gaussian_n5000_d12_s0_p0.5_seed9/'

#file_path = '../../data/sythetic/quad_lingam_n5000_d12_s1_p0.5_seed8/'
#file_path = '../../data/sythetic/quad_gaussian_n5000_d12_s1_p0.5_seed8/'
#file_path = '../../data/sythetic/quad_lingam_n5000_d12_s0_p0.5_seed9/'
#file_path = '../../data/sythetic/quad_gaussian_n5000_d12_s0_p0.5_seed9/'
#'../../data/Heinze-Deml/graph_01'


#file_path = '../../data/flow_cytometric/all/'

# load data

data = np.load(args.data + 'data.npy')
ground_truth = np.load(args.data + 'DAG.npy')

#print(data.shape)
#print(ground_truth.shape)

n = data.shape[0]
d = data.shape[1]

print('Attributes: {}'.format(d))
print('Using {} samples out of {}'.format(n,data.shape[0]))

#var_names = ['praf', 'pmek', 'plcg', 'pip2', 'pip3', 'p44/42', 'pakts473', 'PKA', 'PKC', 'P38', 'pjnk']
var_names = ['x{}'.format(i) for i in range(d)]
X = pd.DataFrame(data, columns=var_names)
#print(X.head())

if not os.path.exists(args.save):
  os.makedirs(args.save)

model = lingam.DirectLiNGAM()

model.fit(X)

res_DAG = model.adjacency_matrix_

print('Finished running model.  Processing results...\n')

for i,j in np.ndindex(d,d):
  if res_DAG[i][j] != 0:
    res_DAG[i][j] = 1
  else:
    res_DAG[i][j] = 0

  if ground_truth[i][j] != 0:
    ground_truth[i][j] = 1
  else:
    ground_truth[i][j] = 0

#print()

#performance = getAllMetrics(res_DAG, ground_truth)
#for k in performance.keys():
#  print(k,':\t', performance[k])

np.save('{}/prediction.npy'.format(args.save), np.array(res_DAG))
np.save('{}/ground_truth.npy'.format(args.save), np.array(ground_truth))

print('Saved.\n')

'''
drawGraph(res_DAG, var_names)
drawGraph(ground_truth, var_names)
#'''
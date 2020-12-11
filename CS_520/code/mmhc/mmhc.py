import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch, BicScore, K2Score
import os

fname = 'dataset_name'
dataFile = '../../data/Heinze-Deml/{}/'.format(fname)

# =====================================================================================

letters_list = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

def to_adjascency_mat(edges, numOfNodes):
    
    m = np.zeros((numOfNodes, numOfNodes))
    
    for e in edges:
        i, j = e[0], e[1]
        m[i-1,j-1] = 1
    return m

def alpha_to_num(L):
    letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    for i in range(len(letters)):
        if L == letters[i]:
            return i
    return -1

def conv_edges(model):
  l = model.edges()
  return [ (alpha_to_num(x),alpha_to_num(y)) for x,y in l ]

# =====================================================================================

dim = 5
data_np = np.load(dataFile+'data.npy')
df = pd.DataFrame(data = data_np)

df = ((df-df.min())/(df.max()-df.min())).astype(np.float32)

truth = np.load(dataFile+'DAG.npy')

df.columns = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[:dim]
print(df.head())
print(df.shape)

# RUNN MMHC on data
est = HillClimbSearch(df, scoring_method=BicScore(df))
best_model = est.estimate(max_indegree=3, max_iter = 10)

# Turn resulting edges to numpy for storage.
m = to_adjascency_mat(conv_edges(best_model), dim)
np.save(f'./output/{fname}/pred', m)

print("Discovered matrix:\n", m)
print(type(m), m.dtype, m.shape)
print(truth!=0)

np.save(f'./output/{fname}/true', np.array(truth))


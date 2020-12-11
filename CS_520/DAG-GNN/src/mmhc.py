import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch, BicScore, K2Score
import os




def to_adjascency_mat(edges, numOfNodes):
    
    m = np.zeros((numOfNodes, numOfNodes))
    
    for e in edges:
        i, j = e[0], e[1]
        m[i-1,j-1] = 1
    return m


# =====================================================================================
dataFile = '../Data/Sachs/1. cd3cd28.xls'

df = pd.read_excel(dataFile)
df.columns = [i for i in range(0, 11)] # REname columns to numbers for easy of use and readability


# RUNN MMHC on data
est = HillClimbSearch(df, scoring_method=K2Score(df),)
best_model = est.estimate( max_iter = 500)

# Turn resulting edges to numpy for storage.
m = to_adjascency_mat(best_model.edges(), 11)
saveFileBase = './Results/mmhc_sachs_data_1'
np.save(saveFileBase+'.npy', m)
np.savetxt(saveFileBase+'.txt', m)

print("Discovered matrix:\n", m)
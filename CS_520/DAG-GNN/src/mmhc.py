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
# Crate target filename and Load data
#dataFile = '../Data/Sachs/1. cd3cd28.xls'
origFolder = 'Small_Datasets'
genProc = 'd5_Gauss_01'
dataFile = os.path.join('..', 'Data', origFolder, genProc, 'data.npy')
fileType = dataFile.split('.')[-1]
# Use appropriate laoding procedure according to file type
if fileType == 'xls':
    df = pd.read_excel(dataFile)
elif fileType == 'npy' or fileType == 'txt':
    data = np.load(dataFile)
    df = pd.DataFrame(data).astype(np.float32)

# Construct columns and normalize them for convergence speed
df.columns = [i for i in range(0, df.shape[1])] # REname columns to numbers for easy of use and readability
#normalized_df=((df-df.min())/(df.max()-df.min())).astype(np.float32)
#print(normalized_df, df.max())
print(df)


# RUNN MMHC on data
est = HillClimbSearch(df, scoring_method=K2Score(df),)
best_model = est.estimate( max_iter = 500)

# Turn resulting edges to numpy for storage.
m = to_adjascency_mat(best_model.edges(), 11)
#saveFileBase = './Results/mmhc_sachs_data_1'
saveFileBase = './Results
saveFile = os.path.join(saveFileBase, '_'.join(('mmhc', genProc, 'predG')))
np.save(saveFileBase+'.npy', m)
np.savetxt(saveFileBase, m)

print("Discovered matrix:\n", m)
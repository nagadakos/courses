import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch, BicScore, K2Score
import os



def to_adjascency_mat(edges, numOfNodes):
    
    m = np.zeros((numOfNodes, numOfNodes))
    
    for e in edges:
        i, j = e[0], e[1]
        m[i,j] = 1
    return m


def main():
    dataFile = '../Data/Sachs/1. cd3cd28.xls'
    #print(os.getcwd())
    df = pd.read_excel(dataFile)
    df.columns = [i for i in range(0, 11)] # REname columns to numbers for easy of use and readability
    print(df)



    est = HillClimbSearch(df, scoring_method=K2Score(df))
    best_model = est.estimate( max_iter = 40)
    m = to_adjascency_mat(best_model.edges(), 11)
    print(best_model.edges())

    saveMatFile = './Results/mmhc_sachs_data_1.npy'
    saveMatTxt = './Results/mmhc_sachs_1_data.txt'
    np.save(saveMatFile, m)
    np.savetxt(saveMatTxt, m, fmt = '%.3e')
    print(m)

if __name__ == "__main__":
    main()
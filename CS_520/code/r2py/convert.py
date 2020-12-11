import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

from rpy2.robjects.conversion import localconverter

dim = 5

base = importr('base')
base.load("./d5_nonGauss_hidden_03.rda")    
rdf_List = base.mget(base.ls())[0]

data = rdf_List[0]
graph = rdf_List[4][0]

print('-'*5 + 'data' + '-'*5)
print(data)
print(graph)

#quit()

data_df = ro.DataFrame(data)
graph_df = ro.DataFrame(graph)

#print('-'*5 + 'df' + '-'*5)
#print(data_df)
#print(graph_df)


with localconverter(ro.default_converter + pandas2ri.converter):
  data_pd = ro.conversion.rpy2py(data_df)
  graph_pd = ro.conversion.rpy2py(graph_df)

#print('-'*5 + 'pd' + '-'*5)
#print(data_pd.shape, data_pd.head())
#print(graph_pd.shape, graph_pd.head())

data_np = data_pd.to_numpy().reshape(dim,1000).T
graph_np = graph_pd.to_numpy().reshape(dim,dim).T

print('-'*5 + 'np' + '-'*5)
print(data_np.shape, '\n', data_np)
print(graph_np.shape, '\n', graph_np)

np.save('./data.npy', data_np)
np.save('./DAG.npy', graph_np)
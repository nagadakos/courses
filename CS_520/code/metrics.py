import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def getConfusionMatrix(result_DAG, true_DAG):
  confusion_matrix = { 'positive':{'true':0, 'false':0}, 'negative':{'true':0, 'false':0} }

  # Check dimensions
  assert len(result_DAG.shape) == 2
  assert len(true_DAG.shape) == 2
  assert result_DAG.shape[0] == true_DAG.shape[0]
  assert result_DAG.shape[1] == true_DAG.shape[1]
  assert true_DAG.shape[0] == true_DAG.shape[1]

  d = true_DAG.shape[0]

  for i,j in np.ndindex(d,d):
    if true_DAG[i][j] == 0:
      if result_DAG[i][j] == 0:
        confusion_matrix['negative']['true'] = confusion_matrix['negative']['true'] + 1
      else:
        confusion_matrix['positive']['false'] = confusion_matrix['positive']['false'] + 1
    else:
      if result_DAG[i][j] == 0:
        confusion_matrix['negative']['false'] = confusion_matrix['negative']['false'] + 1
      else:
        confusion_matrix['positive']['true'] = confusion_matrix['positive']['true'] + 1

  return confusion_matrix

def getPrecision(cm):
  tp = float(cm['positive']['true'])
  fp = float(cm['positive']['false'])
  if tp + fp == 0:
    return 0
  else:
    return tp / (tp + fp)

def getRecall(cm):
  tp = float(cm['positive']['true'])
  fn = float(cm['negative']['false'])
  if tp + fn == 0:
    return 0
  else:
    return tp / (tp + fn)

def getF1score(cm):
  prec = getPrecision(cm)
  rec = getRecall(cm)
  if prec + rec == 0:
    return 0
  else:
    return 2 * prec * rec / (prec + rec)

def getAccuracy(cm):
  t = float(cm['positive']['true'] + float(cm['negative']['true']))
  f = float(cm['positive']['false'] + float(cm['negative']['false']))

  if t + f == 0:
    return 0
  else:
    return t / (t + f)

def getAllMetrics(result_DAG, true_DAG):
  cm = getConfusionMatrix(result_DAG, true_DAG)
  return {'Accuracy': getAccuracy(cm), 'F1 Score': getF1score(cm), 'Precision': getPrecision(cm), 'Recall': getRecall(cm)}

def make_label_dict(labels):
    l = {}
    for i, label in enumerate(labels):
        l[i] = label
    return l

def drawGraph(adj_matrix, var_names):
    positions = {
      1 : [ 1.00000000000000000000 , 0.00000000270930202056 ],
      0 : [ 0.84125351991841190724 , 0.54064077116879127871 ],
      3 : [ 0.41541508156919976225 , 0.90963196532492218704 ],
      2 : [ -0.14231483019311413907 , 0.98982143136684763718 ],
      4 : [ -0.65486066636789563855 , 0.75574964146316248037 ],
      5 : [ -0.95949297081615181337 , 0.28173259018912866214 ],
      7 : [ -0.95949297081615181337 , -0.28173255496820232002 ],
      6 : [ -0.65486072597254008087 , -0.75574957643991402811 ],
      8 : [ -0.14231500900704749379 , -0.98982142594824373827 ],
      10 : [ 0.41541511137152198341 , -0.90963195990631828813 ],
      9 : [ 0.84125346031376746492 , -0.54064088495947626445 ]
    }

    d = adj_matrix.shape[0]

    rows, cols = np.where(adj_matrix != 0)
    edges = zip(rows.tolist(), cols.tolist())
    g = nx.DiGraph()
    g.add_edges_from(edges)
    #pos = nx.drawing.layout.spring_layout(g)
    #pos = nx.drawing.layout.random_layout(g)
    pos = nx.drawing.layout.circular_layout(g)

    #print('pos ({}):'.format(len(pos.keys())))
    #for k in pos.keys():
    #  print('\t',k,': [', f'{pos[k][0]:.20f}',',',f'{pos[k][1]:.20f}','],')  

    for i in range(adj_matrix.shape[0]):
      if i not in pos.keys():
        pos[i] = [-10,-10]

    #if d != 11:
    positions = pos
    nx.draw(g, positions, node_size=500, labels=make_label_dict(var_names), with_labels=True)
    plt.show()

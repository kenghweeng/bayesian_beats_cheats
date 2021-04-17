import numpy as np
import pandas as pd
import LBP

graph = LBP.Graph()
nodes = pd.read_csv('node_potential.csv')
edges = pd.read_csv('edge_potential.csv')

for i, node in nodes.iterrows():
    if np.isnan(node['y']):
        neg, pos = 1 - float(node['1']), float(node['1'])
    elif node['y'] == 0:
        neg, pos = 1 - float(node['1']), 0
    elif node['y'] == 1:
        neg, pos = 0, float(node['1'])
    else:
        print('row %i label not applicable')
        continue
    node_id = str(node['name'])
    graph.rv(node_id, 2, ['0', '1'])
    graph.factor([node_id], potential=np.array([neg, pos]))

for i, edge in edges.iterrows():
    node1_id, node2_id = str(edge['NodeID1']), str(edge['NodeID2'])
    graph.factor([node1_id, node2_id], potential=np.array([[float(edge['0']), float(edge['1']) / 2],
                                                           [float(edge['1']) / 2, float(edge['2'])]]))

i, converged = graph.lbp(normalize=True)
print('{} iterations, converged = {}'.format(i, converged))

marginals = graph.rv_marginals(normalize=True)
df = pd.DataFrame([[i[0], i[1][1]] for i in marginals], columns=['name', 'y_pred'])
df.to_csv('marginals.csv', index=False)
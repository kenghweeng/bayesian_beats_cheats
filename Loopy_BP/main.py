import numpy as np
import pandas as pd
import LBP

graph = LBP.Graph()
node_type, edge_type = 'gpc', 'svm'
nodes = pd.read_csv('%s_node_potential.csv' % node_type)
edges = pd.read_csv('%s_edge_potential.csv' % edge_type)

if edge_type == 'knn':
    MIN = 0.005
else:
    MIN = 0

for i, node in nodes.iterrows():
    if np.isnan(node['y_obs']):
        neg, pos = 1 - float(node['1']), float(node['1'])
    elif node['y_obs'] == 0:
        neg, pos = 1 - float(node['1']), 0
    elif node['y_obs'] == 1:
        neg, pos = 0, float(node['1'])
    else:
        print('row %i label not applicable')
        continue
    node_id = str(node['name'])
    graph.rv(node_id, 2, ['0', '1'])
    graph.factor([node_id], potential=np.array([neg, pos]))

for i, edge in edges.iterrows():
    node1_id, node2_id = str(edge['NodeID1']), str(edge['NodeID2'])
    graph.factor([node1_id, node2_id], potential=np.array([[max(float(edge['0']), MIN),
                                                            max(float(edge['1']) / 2, MIN)],
                                                           [max(float(edge['1']) / 2, MIN),
                                                            max(float(edge['2']), MIN)]]))

i, converged = graph.lbp(normalize=True)
print('{} iterations, converged = {}'.format(i, converged))

marginals = graph.rv_marginals(normalize=True)
df = pd.DataFrame([[i[0], i[1][1]] for i in marginals], columns=['name', 'y_pred'])
df.to_csv('%s_%s_marginals.csv' % (node_type, edge_type), index=False)
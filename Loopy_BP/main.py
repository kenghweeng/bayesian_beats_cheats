import numpy as np
import pandas as pd
import LBP

graph = LBP.Graph()
nodes = pd.read_csv('node_potential.csv')
edges = pd.read_csv('edge_potential.csv')

POTENTAIL_SMOOTHING = 0.005

for i, node in nodes.iterrows():
    if np.isnan(node['y_obs']):
        neg, pos = max(1 - float(node['1']), POTENTAIL_SMOOTHING), \
                   max(float(node['1']), POTENTAIL_SMOOTHING)
    elif node['y_obs'] == 0:
        neg, pos = max(1 - float(node['1']), POTENTAIL_SMOOTHING), 0
    elif node['y_obs'] == 1:
        neg, pos = 0, max(float(node['1']), POTENTAIL_SMOOTHING)
    else:
        print('row %i label not applicable')
        continue
    node_id = str(node['name'])
    graph.rv(node_id, 2, ['0', '1'])
    # print(i, node_id, '\t', node['y_obs'], neg, pos)
    graph.factor([node_id], potential=np.array([neg, pos]))

for i, edge in edges.iterrows():
    node1_id, node2_id = str(edge['NodeID1']), str(edge['NodeID2'])
    graph.factor([node1_id, node2_id], potential=np.array([[max(float(edge['0']), POTENTAIL_SMOOTHING),
                                                            max(float(edge['1']) / 2, POTENTAIL_SMOOTHING)],
                                                           [max(float(edge['1']) / 2, POTENTAIL_SMOOTHING),
                                                            max(float(edge['2']), POTENTAIL_SMOOTHING)]]))

i, converged = graph.lbp(normalize=True)
print('{} iterations, converged = {}'.format(i, converged))
# graph.print_rv_marginals()
marginals = graph.rv_marginals(normalize=True)
df = pd.DataFrame([[i[0], i[1][1]] for i in marginals], columns=['name', 'y_pred'])
df.to_csv('marginals.csv', index=False)
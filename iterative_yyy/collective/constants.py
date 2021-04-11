from networkx import Graph
import pandas as pd


def get_summary_zv(networkx_graph: Graph):
    """
    :param: networkx_graph: networkx's Graph
    """
    # zv = {}
    for node in networkx_graph.nodes:
        L1_max = 0
        L0_max = 0
        L1 = []
        L0 = []
        neighbors = networkx_graph.neighbors(node)
        for neighbor in neighbors:
            neighbor_node = networkx_graph.nodes[neighbor]
            edge_weight = networkx_graph[neighbor][node]['edge_weight']
            if (neighbor_node['label'] == 1):
                L1_max = max(L1_max, edge_weight)
                L1 = L1 + [edge_weight]
            else:
                L0_max = max(L0_max, edge_weight)
                L0 = L0 + [edge_weight]
        networkx_graph.nodes[node]['L1_max'] = L1_max
        networkx_graph.nodes[node]['L0_max'] = L0_max
        networkx_graph.nodes[node]['L1_mean'] = sum(
            L1) / len(L1) if len(L1) != 0 else 0
        networkx_graph.nodes[node]['L0_mean'] = sum(
            L0) / len(L0) if len(L0) != 0 else 0
    return networkx_graph


def get_model1_features(node_attr, node):
    node_attr.pop('label', None)
    node_attr.pop('L1_max', None)
    node_attr.pop('L0_max', None)
    node_attr.pop('L1_mean', None)
    node_attr.pop('L0_mean', None)
    node_attr['index'] = node
    return pd.DataFrame([node_attr]).set_index('index')


def get_model2_features(node_attr, node):
    node_attr.pop('label', None)
    node_attr['index'] = node
    return pd.DataFrame([node_attr]).set_index('index')

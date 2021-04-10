from networkx import DiGraph
import pandas as pd


def get_summary_zv(networkx_graph: DiGraph):
    """
    :networkx_graph: networkx's DiGraph
    """
    # zv = {}
    for node in networkx_graph.nodes:
        i0 = 0
        i1 = 0
        o0 = 0
        o1 = 0
        predecessors = networkx_graph.predecessors(node)
        for predecessor in predecessors:
            if (i0 == 1 and i1 == 1):
                break
            if (networkx_graph.nodes[predecessor]['label'] == 0):
                i0 = 1
            elif (networkx_graph.nodes[predecessor]['label'] == 1):
                i1 = 1
        successors = networkx_graph.successors(node)
        for successor in successors:
            if (o0 == 1 and o1 == 1):
                break
            if (networkx_graph.nodes[successor]['label'] == 0):
                o0 = 1
            elif (networkx_graph.nodes[successor]['label'] == 1):
                o1 = 1
        # zv[node] = [i0, i1, o0, o1]
        networkx_graph.nodes[node]['i0'] = i0
        networkx_graph.nodes[node]['i1'] = i1
        networkx_graph.nodes[node]['o0'] = o0
        networkx_graph.nodes[node]['o1'] = o1
    return networkx_graph


def get_model1_features(node_attr, node):
    node_attr.pop('label', None)
    node_attr.pop('i0', None)
    node_attr.pop('i1', None)
    node_attr.pop('o0', None)
    node_attr.pop('o1', None)
    node_attr['index'] = node
    return pd.DataFrame([node_attr]).set_index('index')


def get_model2_features(node_attr, node):
    node_attr.pop('label', None)
    node_attr['index'] = node
    return pd.DataFrame([node_attr]).set_index('index')

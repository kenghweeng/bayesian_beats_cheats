import pandas as pd
import networkx as nx

def create_nx_graph_nodes(x):
    network_graph = nx.Graph()
    for index, row in x.iterrows():
        network_graph.add_node(row['name'])
        network_graph.nodes[row['name']].update(row.drop(['name']).to_dict())
    return network_graph

def add_nx_graph_edges(network_graph, df_edge):
    for node in network_graph.nodes:
        edge1 = df_edge[df_edge.NodeID1 == node]
        edge2 = df_edge[df_edge.NodeID2 == node]
        if (len(edge1) != 0):
            for index, edge in edge1.iterrows():
                if (edge.NodeID2 in network_graph.nodes):
                    network_graph.add_edge(node, edge.NodeID2)
                    network_graph[node][edge.NodeID2]['edge_weight'] = edge.edge_weights
        elif (len(edge2) != 0):
            for index, edge in edge2.iterrows():
                if (edge.NodeID1 in network_graph.nodes):
                    network_graph.add_edge(node, edge.NodeID1)
                    network_graph[node][edge.NodeID1]['edge_weight'] = edge.edge_weights
    return network_graph
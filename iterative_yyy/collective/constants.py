from src import preprocess
from networkx import Graph
import networkx as nx
import pandas as pd
import os
from pathlib import Path
while Path.cwd().name != 'bayesian_beats_cheats':
    os.chdir(Path.cwd().parent)


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


def add_nodes_edges_to_graph(df_node, df_edge):
    # add nodes to graph
    network_graph = create_nx_graph_nodes(df_node)
    # add edges to graph
    network_graph = add_nx_graph_edges(network_graph, df_edge)
    # compute L1_max, L0_max, L1_mean, L0_mean
    network_graph = get_summary_zv(network_graph)
    # convert to df
    df_final = pd.DataFrame.from_dict(network_graph.nodes, orient='index')
    return [network_graph, df_final]


def get_prec_at_k(df_test, model2, y_test, k=20):
    df_final = df_test.copy()
    predictions = model2.predict_proba(df_final.drop(['label'], axis=1))[:, 1]
    df_final['true_labels'] = y_test.values
    df_final['pred_label'] = predictions
    df_final.sort_values('pred_label', ascending=False, inplace=True)
    return df_final.head(k).true_labels.sum() / k


def get_df(node_file_path, edge_file_path, clip=True):
    df_node = pd.read_csv(node_file_path, keep_default_na=False)
    df_edge = pd.read_csv(edge_file_path)

    # combine train and val together to train
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess.stratified_train_val_test_split(df_node)

    for x_df in [X_train, X_val, X_test]:
        x_df.drop(columns=['confessed_assignments'], inplace=True)
    for y_df in [y_train, y_val, y_test]:
        y_df.update((y_df > 0).astype(int))
        y_df.rename("label", inplace=True)

    # check class distribution
    print("Class distributions: ")
    print(y_train.value_counts())
    print()

    # Clipping max edge weights
    if clip:
        print("Clipping edge weights and num_videos")
        df_edge.loc[df_edge.edge_weights > 0.05, "edge_weights"] = df_edge[df_edge.edge_weights > 0.05].edge_weights.min()
        X_train.loc[X_train.num_videos > 50, "num_videos"] = X_train[X_train.num_videos > 50].num_videos.min()
        X_val.loc[X_val.num_videos > 50, "num_videos"] = X_val[X_val.num_videos > 50].num_videos.min()
        X_test.loc[X_test.num_videos > 50, "num_videos"] = X_test[X_test.num_videos > 50].num_videos.min()

    return [X_train, y_train, df_edge, X_val, y_val, X_test, y_test]

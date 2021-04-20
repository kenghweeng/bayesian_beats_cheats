# Assume phi_1 and phi_2 has already been trained, and are model1 and model2 respectively.
from networkx import DiGraph
from collective.constants import get_summary_zv, get_model1_features, get_model2_features
import pandas as pd
import numpy as np


class IterativeClassification:
    __max_iterations = 1000

    def __init__(self, max_iterations=__max_iterations):
        self.__max_iterations = max_iterations

    def iterate(self, networkx_graph: DiGraph, model1, model2, df):
        old_labels = -np.ones(df['label'].shape, dtype=np.int)
        num_iter = 0
        while np.sum(df['label'] != old_labels) > 0 and num_iter < self.__max_iterations:
            num_iter += 1
            old_labels = df['label'].values.copy()

            df['label'] = model2.predict(df.drop(['label'], axis=1))
            for name, label in df['label'].iteritems():
                networkx_graph.nodes[name]['label'] = label
            networkx_graph = get_summary_zv(networkx_graph)  # update L0, L1 features
            df = pd.DataFrame.from_dict(networkx_graph.nodes, orient='index')

        if num_iter == self.__max_iterations:
            print(f"did not converge in {self.__max_iterations}")
        else:
            print(f"converged in {num_iter} iterations")
        return [networkx_graph, df]

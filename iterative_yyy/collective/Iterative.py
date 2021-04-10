# Assume phi_1 and phi_2 has already been trained, and are model1 and model2 respectively.
from networkx import DiGraph
from collective.constants import get_summary_zv, get_model1_features, get_model2_features


class IterativeClassification:
    __max_iterations = 1000

    def __init__(self, max_iterations=__max_iterations):
        self.__max_iterations = max_iterations

    def predict(self, networkx_graph: DiGraph, model1, model2):
        classes_changed = True
        iterations = 0
        # ori_graph = networkx_graph.copy()

        # Step 2
        print("In step 2 of iterative classification")
        for node in networkx_graph.nodes:
            networkx_graph.nodes[node]['label'] = model1.predict(
                get_model1_features(networkx_graph.nodes[node], node))

        # Step 3
        print("In step 3 of iterative classification")
        while(iterations < self.__max_iterations and classes_changed):
            print("At iteration " + str(iterations))
            if iterations % 100 == 0:
                print("At iteration " + str(iterations))
            classes_changed = False
            prev_graph = networkx_graph.copy()
            networkx_graph = get_summary_zv(networkx_graph)
            for node in networkx_graph.nodes:
                networkx_graph.nodes[node]['label'] = model2.predict(
                    get_model2_features(networkx_graph.nodes[node], node))
                if (networkx_graph.nodes[node]['label'] != prev_graph.nodes[node]['label']):
                    classes_changed = True
            iterations = iterations + 1
        return networkx_graph

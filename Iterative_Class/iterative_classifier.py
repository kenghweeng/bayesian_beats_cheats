from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# https://web.stanford.edu/class/cs224w/slides/06-collective.pdf

class IterativeClassifier():
    def __init__(self, n_neighbors=3):
        self.clf_feature = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.clf_feature_link = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, nodes, y):
        """
        On different training set, train two classifiers:
            1. Word vector only
            2. Word and link vectors 
        """
        feature_vector = generate_feature_vector(nodes)
        feature_link_vector = generate_feature_link_vector(nodes, y)
        self.clf_feature.fit(feature_vector, y)
        self.clf_feature_link.fit(feature_link_vector, y)

    def predict(self, nodes):

        # Bootstrap: use trained word-vector classifier to bootstrap on test set
        feature_vector = generate_feature_vector(nodes)
        y = self.clf_feature.predict(feature_vector)

        # Iterate until convergence:
        # 1. Update neighborhood vector for all nodes
        # 2. Reclassify all nodes 
        feature_link_vector = generate_feature_link_vector(nodes, y)
        while True:
            y_new = self.clf_feature_link.predict(feature_link_vector)
            comparison = np.array(y) == np.array(y_new)
            y = y_new
            if comparison.all():
                break
            feature_link_vector = generate_feature_link_vector(nodes, y)

        return y

    def generate_feature_vector(nodes):
        # TODO
        return []

    def generate_feature_link_vector(nodes, y):
        # TODO
        return []
        

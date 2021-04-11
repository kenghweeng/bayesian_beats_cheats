from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import sys

# https://web.stanford.edu/class/cs224w/slides/06-collective.pdf

class IterativeClassifier():

    def __init__(self, n_neighbors=3):
        self.clf_feature = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.clf_feature_link = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, fetures, edge_weights, y):
        """
        On different training set, train two classifiers:
            1. Word vector only
            2. Word and link vectors 
        """
        X_feature = self.generate_feature_vector(fetures)
        X_feature_link = self.generate_feature_link_vector(fetures, edge_weights, y)
        self.clf_feature.fit(X_feature, y)
        self.clf_feature_link.fit(X_feature_link, y)

    def predict(self, fetures, edge_weights):

        # Bootstrap: use trained word-vector classifier to bootstrap on test set
        X_feature = self.generate_feature_vector(fetures)
        y = self.clf_feature.predict(X_feature)

        # Iterate until convergence:
        # 1. Update neighborhood vector for all nodes
        # 2. Reclassify all nodes 
        X_feature_link = self.generate_feature_link_vector(fetures, edge_weights, y)
        while True:
            y_new = self.clf_feature_link.predict(X_feature_link)
            comparison = np.array(y) == np.array(y_new)
            y = y_new
            if comparison.all():
                break
            X_feature_link = self.generate_feature_link_vector(fetures, edge_weights, y)

        return y

    def generate_feature_vector(self, features):
        X = []
        for i in range(len(features)):
            X.append(features[i][1:])
        return X

    def generate_feature_link_vector(self, features, edge_weights, y):
        X = []
        
        name_index_dict = {}
        for i in range(len(features)):
            name_index_dict[features[i][0]] = i
            
        for i in range(len(features)):
            L1_max = sys.float_info.min
            L0_max = sys.float_info.min
            L1_avg = 0
            L0_avg = 0
            count1 = 0
            count0 = 0
            
            x = features[i][1:]
            source = features[i][0]
            try:
                weights = edge_weights[source]
            except:
                print('no source {}'.format(source))
            for j in range(len(weights)):
                neighbor = weights[j][0]
                try:
                    index = name_index_dict[neighbor]
                    label = y[index]
                    if label == 1:
                        if weights[j][2] > L1_max:
                            L1_max = weights[j][2]
                        L1_avg += weights[j][1]
                        count1 += 1
                    else:
                        if weights[j][2] > L0_max:
                            L0_max = weights[j][2]
                        L0_avg += weights[j][1]
                        count0 += 1
                except:
                    print('no neighbor {}'.format(neighbor))
                
            try:
                L1_avg /= count1
            except:
                print(count1, features[i][0])
                
            try:
                L0_avg /= count0
            except:
                print(count0, features[i][0])
            
            x.append(L1_max)
            x.append(L0_max)
            x.append(L1_avg)
            x.append(L0_avg)
            X.append(x)
            
        return X

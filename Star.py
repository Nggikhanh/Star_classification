import numpy as np

class Knearestneighbor:
    def __init__(self, trainData, trainLabel, n_neighbors=1, distance_metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric
        self.X_train = trainData
        self.y_train = trainLabel

    def calculated_distance(self, a, b):
        self.distance_metric == 'euclidean'
        return np.sqrt(np.sum((a - b) ** 2, axis=1))  # Use axis=0 for element-wise subtraction
    
        
    def Kneighbors(self, Test_data):
        dist = []
        neigh_ind = []
        point_dist = []
        point_dist = [self.calculated_distance(singleTestData, self.X_train) for singleTestData in Test_data]
        
        for Singletestdata in point_dist:
            enum_neigh = enumerate(Singletestdata)
            sorted_neigh = sorted(enum_neigh, key=lambda x: x[1])[:self.n_neighbors]
            ind_list = [tup[0] for tup in sorted_neigh]
            dist_list = [tup[1] for tup in sorted_neigh]
            dist.append(dist_list)
            neigh_ind.append(ind_list)
            
            #return np.array(neigh_ind, dist)
        return neigh_ind
    
    def predict(self, X_test):
        neighbors = self.Kneighbors(X_test) 
        y_pred = np.array([np.argmax(np.bincount(self.y_train[neighbor])) for neighbor in neighbors])
        return y_pred
    
    def predict_proba(self, X_test):
        neighbors = self.Kneighbors(X_test)
        proba_list = []

        for neighbor in neighbors:
            y_values = self.y_train[neighbor]
            y_count = np.bincount(y_values, minlength=len(set(self.y_train)))
            class_probabilities = y_count / len(neighbor)
            proba_list.append(class_probabilities)

        return np.array(proba_list)
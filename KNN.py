import numpy as np
import statistics

class KNN_classifier():
    #Initiating the parameters
    def __init__(self, distance_metric):
        self.distance_metric = distance_metric

    #getting the distance metric
    def get_distance_metric(self, training_data_point, test_data_point):
        if self.distance_metric == 'Euclidean':
            dist = 0
            for i in range(len(training_data_point) -1): #-1 is there to exclude the values on the target column
                dist = dist + (training_data_point[i] - test_data_point[i])**2
            euclidean_distance = np.sqrt(dist)
            return euclidean_distance
        
        elif self.distance_metric == 'Manhattan':
            dist = 0
            for i in range(len(training_data_point) -1):
                dist = dist + abs(training_data_point[i] - test_data_point[i])
            manhattan_dist = dist
            return manhattan_dist
        else:
            raise ValueError("Unsupported distance metric")

    # Getting the nearest neighbours    
    def nearest_neighbours(self, X_train, test_data, K):
        distance_list = []
        for training_data in X_train:
            distance = self.get_distance_metric(training_data, test_data)
            distance_list.append((training_data, distance))

        distance_list.sort(key = lambda x:x[1])
        
        neighbours_list = []
        for j in range(K):
            neighbours_list.append(distance_list[j][0])
        return neighbours_list

    #predicting the class of the data point
    def predict(self, X_train, test_data, K):
        neighbours = self.nearest_neighbours(X_train, test_data, K)
        
        label =[]
        for data in neighbours:
            label.append(data[-1])
        predicted_class = statistics.mode(label)
        return predicted_class

    #it is important for target to be the last column in KNN
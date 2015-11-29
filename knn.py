
import numpy as np
from pyspark import SparkContext
from utils import eucdist, vote, find_neighbours, get_confusion_matrix

class KNN(object):

    '''
        KNN for PySpark
    '''

    def __init__(self, data, k=1):
        '''
            Initialize a KNN object
            data: RDD Type
            data format: (index, class, features)
        '''
        self.k = int(k)
        self.data = data

    def update_k(self, k):
        self.k = int(k)

    def get_pair_distance(self):
        # Creates key value pair where key is index1
        pairs = self.data.cartesian(self.data).filter(lambda (x1, x2): x1[0] != x2[0])
        #Applies euclidean distance function to all pairs
        imgED = pairs.map(eucdist)
        # Creates key value pair where key is index1
        KVpair = imgED.map(lambda x: (x[0],x[1:]))
        # Group distances for each key
        distRDD = KVpair.groupByKey().mapValues(list)

        return distRDD

    def get_k_nearest_neighbours(self):
        # Find k nearest neighbor for each key
        # for each k, returns a list of tuples; each tuple (neighbour_index, neighbour_label)
        distRDD = self.get_pair_distance()
        k = int(self.k)
        sortedDistRDD = distRDD.map(lambda (idx, arr) : (idx, find_neighbours(arr, k)))
        return sortedDistRDD

    def train(self):
        '''return confusion matrix'''
        sortedDistRDD = self.get_k_nearest_neighbours()
        # Predict -- Majority Voting
        predictionRDD = sortedDistRDD.map(lambda (idx, knns) : (idx, vote(knns)))
        # Get actual labels
        actualClassRDD = self.data.map(lambda (index, cl, features) : (index, cl))

        pred_tuple = predictionRDD.collect()
        true_tuple = actualClassRDD.collect()

        pred, true = np.zeros(len(pred_tuple)), np.zeros(len(pred_tuple)) 
        for i in range(len(pred)):
            # Pred
            idx, cl = pred_tuple[i]
            pred[int(idx)] = int(cl)
            # Actual
            idx, cl =true_tuple[i]
            true[int(idx)] = int(cl)

        confusion_matrix = get_confusion_matrix(pred, true)
        return confusion_matrix

    def predict(self, other_data):
        '''
            Returns a RDD object which stores index and predictions
            @param
            other_data should also be a RDD object;
            but it does NOT have class label for each sample
            each item is stored as (index, features)
        '''
        # Create pair
        pairs = other_data.cartesian(self.data)
        # Applies euclidean distance function to all pairs
        pointED = pairs.map(eucdist)
        # Creates key value pair where key is index1
        KVpair = pointED.map(lambda x: (x[0],x[1:]))
        # Group distances for each key
        distRDD = KVpair.groupByKey().mapValues(list)
        
        # Find k Nearest Neighbours
        k = int(self.k)
        sortedDistRDD = distRDD.map(lambda (idx, arr) : (idx, find_neighbours(arr, k)))
        predictionRDD = sortedDistRDD.map(lambda (idx, knns) : (idx, vote(knns)))
        return predictionRDD

    def test(self, test_data):
        '''test_data should also be a RDD object'''
        predictionRDD = self.predict(test_data)

        # Get actual labels
        actualClassRDD = test_data.map(lambda (index, cl, features) : (index, cl))

        pred_tuple = predictionRDD.collect()
        true_tuple = actualClassRDD.collect()

        pred, true = np.zeros(len(pred_tuple)), np.zeros(len(pred_tuple)) 
        for i in range(len(pred)):
            # Pred
            idx, cl = pred_tuple[i]
            pred[int(idx)] = int(cl)
            # Actual
            idx, cl =true_tuple[i]
            true[int(idx)] = int(cl)

        confusion_matrix = get_confusion_matrix(pred, true)
        return confusion_matrix

if __name__ == '__main__':
    sc = SparkContext()
    # Read file
    img = sc.textFile('./test.txt')
    imgRDD = img.map(lambda s: [int(t) for t in s.split()])
    # Adds the index
    RDDind = imgRDD.zipWithIndex()
    # Switches positions of index and data
    indRDD = RDDind.map(lambda (data,index):(index,data))
    # Organizes into index,class,features
    indClassFeat = indRDD.map(lambda (index,data): (index,data[-1],data[:-1]))

    knn = KNN(indClassFeat)
    knn.train()


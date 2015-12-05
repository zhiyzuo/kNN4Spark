class KNN(object):

    '''
        KNN for PySpark
    '''

    def __init__(self, feature, label, k=1, norm=2, smooth=1, weighted=False):
        '''
            Initialize a KNN object
        '''
        self.k = int(k)
        self.norm = norm
        self.data_feature = feature
        self.data_label = label
        self.smooth = smooth
        self.weighted = weighted

    def change_k(self, k):
        self.k = int(k)

    def change_norm(self, norm):
        self.norm = int(norm)

    def predict(self, other_data, length=None):
        '''
            Returns a RDD object which stores index and predictions
            @param
            other_data should also be a RDD object;
                        but it does NOT have class label for each sample
                        each item is stored as (index, features)

            length is the length of other_data
        '''

        if length == None:
            length = other_data.count()

        # Create pair: each test point is associated with a subgroup of train data
        pairs = other_data.cartesian(self.data).collect()
        # loop through each pair
        for test_idx in range(length):
            # get subset of this test index
            idx_pair = pairs.filter(lambda (testpoint, trainsubgroup): testpoint[0] == test_idx)


        # Applies euclidean distance function to all pairs
        norm = self.norm
        pointED = pairs.map(lambda x: get_distance(x, norm))
        # Creates key value pair where key is index1
        KVpair = pointED.map(lambda x: (x[0],x[1:]))
        # Group distances for each key
        distRDD = KVpair.groupByKey().mapValues(list)
        
        # Find k Nearest Neighbours
        k = int(self.k)
        sortedDistRDD = distRDD.map(lambda (idx, arr) : (idx, find_neighbours(arr, k)))
        weighted, smooth = self.weighted, self.smooth
        predictionRDD = sortedDistRDD.map(lambda (idx, knns) : (idx, vote(knns, weighted, smooth)))
        return predictionRDD

    def test(self, test_data, test_label):
        '''test_data should also be a RDD object'''
        predictionRDD = self.predict(test_data)

        # Get actual labels
        actualClassRDD = test_label

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


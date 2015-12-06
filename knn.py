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
        from utils import cdist,vote

        if length == None:
            length = other_data.count()

        # Create pair: each test point is associated with a subgroup of train data
        pairs = other_data.cartesian(self.data_feature)
        # loop through each pair
        dist_label_tuple_list = []
        predictions = []
        for test_idx in range(length):
            # get subset of this test index; collect to do for loop
            idx_pairs = pairs.filter(lambda (testpoint, trainsubgroup): testpoint[0] == test_idx).collect()
            for idx_p in idx_pairs:
                # find out the train subgroup
                train_subgroup_idx = idx_p[1][0]
                # Their Class
                C = self.data_label.filter(lambda (ind, subgroup): ind == train_subgroup_idx).collect()[0]
                dist_label_tuple_list.extend(cdist(idx_p[0], idx_p[1], C, self.k))
            predictions.append((test_idx, vote(dist_label_tuple_list, self.k)))
            del idx_pairs
 
        return predictions

    def test(self, test_data, test_label):
        '''test_data should also be a RDD object'''

        import numpy as np
        pred_tuple = self.predict(test_data)

        # Get actual labels
        true_tuple = test_label.collect()

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


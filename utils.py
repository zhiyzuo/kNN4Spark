
def get_image_rdd(sc, n_groups=None, start=0, end=10, resize=0.1):
    '''
        Retrieve pixels as RDDs from images
        Also do reshape to n_groups of train data

        Update on 12/05/15: 
            return (index, feature) and (index, label)
            if n_groups != None, return reshaped features and labels
    '''

    import os
    import numpy as np
    from processImage import processImage


    images = os.listdir("../Original/train/")
    imgsRDD = sc.parallelize(images[start:end])
    pixelsRDD = imgsRDD.flatMap(lambda x : processImage(x, resizeTo=resize))

    RDDind = pixelsRDD.zipWithIndex()
    indRDD = RDDind.map(lambda (data,index):(index,data))
    index_feature = indRDD.map(lambda (index,data): (index,data[:-1]))
    index_label = indRDD.map(lambda (index,data): (index,data[-1]))

    if n_groups == None:
        return index_feature, index_label
    
    # Do reshape
    # Do a following map to convert from resultiterable to list
    regroup_feature = index_feature.groupBy(lambda x: x[0] % n_groups).map(lambda (idx, x): (idx, sorted(x)))
    regroup_label = index_label.groupBy(lambda x: x[0] % n_groups).map(lambda (idx, x): (idx, sorted(x)))
    return regroup_feature, regroup_label

def get_confusion_matrix(pred, true, see=True):
    import numpy as np
    import numpy.matlib
    '''
        Return confusion matrix
    '''
    pred, true = np.asarray(pred), np.asarray(true)
    confusion_matrix = numpy.matlib.zeros((2,2), dtype=int)
    for i in range(len(pred)):
        if pred[i] == true[i]:
            if pred[i] == 1: # True Positive
                confusion_matrix[0,0] += 1
            else: # True Negative
                confusion_matrix[1,1] += 1
        else:
            if pred[i] == 1: # False Positive
                confusion_matrix[0,1] += 1
            else: # False Negative
                confusion_matrix[1,0] += 1

    TP, TN, FP, FN = confusion_matrix[0,0], confusion_matrix[1,1], confusion_matrix[0,1], confusion_matrix[1,0]

    if see:
        output = '\t  Actual\n\t+\t-\nPred + %d\t%d\n     - %d\t%d' %(TP, FP, FN, TN)
        print output

    return confusion_matrix

def vote (DCpairs, k):
    '''
        Sorts distance, class pairs
        Returns majority class given k
    '''
    import numpy as np
    
    DCpairs = np.asarray(DCpairs)
    sort_dist = DCpairs[DCpairs[:,0].argsort()]
    k_entries = sort_dist[0:k]
    kClsList = list(k_entries[:,1])
    return int(max(set(kClsList), key=kClsList.count))
    
    
#def vote(knns, weighted=False, smooth=1):
 #   '''
  #      Return predictions by voting
   #     Default: Equal weights for all neighbours
    #'''
#
 #   # binary class
  #  pred_dict = {0: 0., 1:0.}
   # for n_i in knns:
    #    this_label = int(n_i[1])
#
 #       if weighted:
  #          # weight inversely proportional to distance
   #         distance = float(n_i[-1])
    #        # separate distance 0 due to ZeroDivisionError
     #       # Smooth denominator to avoid zero denominator
      #      pred_dict[this_label] += 1./(distance+smooth)
#
 #       else:
  #          pred_dict[this_label] += 1
#
 #   # return the majority class label
  #  return sorted(pred_dict, key=pred_dict.get)[-1]

def cdist(u, A, C, k, norm=2):
    '''
        Calcualte the distance between
        vector u and matrix A with norm = 2
        
        @param
        u: (index, feature);
        A: (index, [(index, feature)])
        C: (index, [(index, class)])
        k: # of nearest neighoubour
    '''

    # [(index, distance)]
    distance_vector = dist(u, A)
    smallest_k = sorted(distance_vector, key=lambda x: x[1])[:k]

    for item in smallest_k:
        # find its class by index (1st element)
        cls = C[idx][item[0]]
        result.append((item[1], cls))

    return result

def dist(u, A, norm=2):
    '''
        Distance between u and A

        @param
        u: (index, feature)
        A: (index, [(index, feature)])
    '''

    import numpy as np
    distance_vector = []

    # 2nd element of A is a list of training samples
    for idx in range(len(A[1])):
        v = A[1][idx]
        # append (vi, distance)
        distance_vector.append(dist_(u, v, norm))
    return distance_vector

def dist_(u, v, norm=2):
    '''
        Calculate distance between u and v

        @param
        u, v: (index, feature)

        ** return the second index
    '''
    import numpy as np

    ui, vi = u[0], v[0]
    uf, vf = u[1], v[1]
    d = np.linalg.norm(np.asarray(uf) - np.asarray(vf), norm)
    # cast to 16-bit floating point
    return tuple([vi, np.float16(d)])

'''
def get_distance(p, norm=2):
    import numpy as np
    p1, p2 = p
    #Unpacks first point into index,class,features

    # Distinguish test and train
    if len(p1) < 3:
        i1,f1 = p1
    else:
        i1,c1,f1=p1
    #Unpacks second point into index,class,features
    i2,c2,f2=p2
    #Initializes distance
    dist = np.linalg.norm(np.asarray(f1) - np.asarray(f2), norm)
    #Returns index1,index2,class1,class2,euclidean distance
    if len(p1) < 3:
        return (i1,i2,c2,dist)
    else:
        return (i1,i2,c1,c2,dist)

def find_neighbours(dist_array, k=1):

    import numpy as np
    ind_dist_dict = {}
    info_dict = {}
    for item in dist_array:
        ind_dist_dict[item[0]] = item[-1]
        info_dict[item[0]] = item[1:]
    top_k_indices = sorted(ind_dist_dict, key=ind_dist_dict.get)[:k]
    
    # return near neighbours along with their labels

    # distinguish train and test
    if len(dist_array[0]) < 4:
        label_index = 0
    else:
        label_index = 1

    k_neighbors = [(idx, info_dict[idx][label_index], info_dict[idx][-1]) for idx in top_k_indices]

    return k_neighbors

'''

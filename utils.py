def slice_list(data_list, num):
    '''
        Collects the output of processImage and splits
    '''
    avg = len(data_list) / float(num)
    slices = []
    position = 0.0
    while position < len(data_list):
        slices.append(data_list[int(position):int(position + avg)])
        position += avg
    return slices


def get_image_rdd(sc, n_=100, resize=0.1):
    '''
        Retrieve pixels as RDDs from images

        Update on 12/05/15: return (index, feature) and (index, label)
    '''

    import os
    from processImage import processImage


    images = os.listdir("../Original/train/")
    imgsRDD = sc.parallelize(images[:n_])
    pixelsRDD = imgsRDD.flatMap(lambda x : processImage(x, resizeTo=resize))

    RDDind = pixelsRDD.zipWithIndex()
    indRDD = RDDind.map(lambda (data,index):(index,data))
    index_feature = indRDD.map(lambda (index,data): (index,data[:-1]))
    index_label = indRDD.map(lambda (index,data): (index,data[-1]))

    return index_feature, index_label


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

def vote(knns, weighted=False, smooth=1):
    '''
        Return predictions by voting
        Default: Equal weights for all neighbours
    '''

    # binary class
    pred_dict = {0: 0., 1:0.}
    for n_i in knns:
        this_label = int(n_i[1])

        if weighted:
            # weight inversely proportional to distance
            distance = float(n_i[-1])
            # separate distance 0 due to ZeroDivisionError
            # Smooth denominator to avoid zero denominator
            pred_dict[this_label] += 1./(distance+smooth)

        else:
            pred_dict[this_label] += 1

    # return the majority class label
    return sorted(pred_dict, key=pred_dict.get)[-1]

def find_neighbours(dist_array, k=1):
    '''
        Find the closest neighbours and 
        return their indices, class labels, and distances (for weighted voting)
    '''

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

def get_distance(p, norm=2):

    '''
        Euclidean distance function, takes in a pair of points
    '''

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

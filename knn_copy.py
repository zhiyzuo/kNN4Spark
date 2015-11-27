def knn(data, k=3):

    import numpy as np
    from pyspark import SparkContext
    from utils import eucdist, vote, find_neighbours

    sc = SparkContext()

    # Read file
    img = sc.textFile(data)
    imgRDD = img.map(lambda s: [int(t) for t in s.split()])

    # Adds the index
    RDDind = imgRDD.zipWithIndex()

    # Switches positions of index and data
    indRDD = RDDind.map(lambda (data,index):(index,data))

    # Organizes into index,class,features
    indClassFeat = indRDD.map(lambda (index,data): (index,data[-1],data[:-1]))

    # Index, Class; Index, Feature
    indClass = indClassFeat.map(lambda (index, cl, features) : (index, cl))
    indFeat = indClassFeat.map(lambda (index, cl, features) : (index, features))

    # Creates all pairs of points
    # Filter out the self-self pairs 
    pairs = indClassFeat.cartesian(indClassFeat).filter(lambda (x1, x2): x1[0] != x2[0])

    #Applies euclidean distance function to all pairs
    imgED = pairs.map(eucdist)

    # Creates key value pair where key is index1
    KVpair = imgED.map(lambda x: (x[0],x[1:]))

    # Group distances for each key
    distRDD = KVpair.groupByKey().mapValues(list)

    # Find k nearest neighbor for each key
    # for each k, returns a list of tuples; each tuple (neighbour_index, neighbour_label)
    sortedDistRDD = distRDD.map(lambda (idx, arr) : (idx, find_neighbours(arr)))

    # Predict -- Majority Voting
    predictionRDD = sortedDistRDD.map(lambda (idx, knns) : (idx, vote(knns)))

if __name__ == '__main__':
    knn('./test.txt')



import numpy as np
from knn import KNN
from pyspark import SparkConf, SparkContext
from utils import get_distance, vote, find_neighbours, get_confusion_matrix, get_image_rdd

sc = SparkContext()

indClassFeat = get_image_rdd(sc, n_=1000)

flatten = indClassFeat.flatMap(lambda x:x)
flatten_list = flatten.collect()
data_slices = slice_list(flatten_list,100)
slicesRDD = sc.parallelize(data_slices)

knn = KNN(indClassFeat)
#knn.loo()

sc.stop()


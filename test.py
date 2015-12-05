
import numpy as np
from knn import KNN
from pyspark import SparkConf, SparkContext
from utils import cdist, vote, get_confusion_matrix, get_image_rdd

sc = SparkContext()

# train data
# Each element in x and y is (SubGroupKey, iterableResults)
# in which iterableResults are (PixelKey, features/labels)
x, y = get_image_rdd(sc, n_groups=5, start=0, end=10)
x_, y_ = get_image_rdd(sc, start=10, end=11)

knn = KNN(x,y)

sc.stop()

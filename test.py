
import numpy as np
from knn import KNN
from pyspark import SparkConf, SparkContext
from utils import get_distance, vote, find_neighbours, get_confusion_matrix, get_image_rdd

sc = SparkContext()

# train data
# Each element in x and y is (SubGroupKey, iterableResults)
# in which iterableResults are (PixelKey, features/labels)
x, y = get_image_rdd(sc, n_groups=1000, start=0, end=1300)

# test data: read one image every time
# use for loop
# Try one image first
x_, y_ = get_image_rdd(sc, start=1300, end=2000)

sc.stop()


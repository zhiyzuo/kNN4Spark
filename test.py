
import numpy as np
from knn import KNN
from pyspark import SparkConf, SparkContext
from utils import get_distance, vote, find_neighbours, get_confusion_matrix, get_image_rdd

sc = SparkContext()

x, y = get_image_rdd(sc, n_groups=5, n_=10)

sc.stop()


import numpy as np
import numpy.matlib
from knn import KNN
from pyspark import SparkConf, SparkContext
from utils import cdist, vote, get_confusion_matrix, get_image_rdd

sc = SparkContext()

# train data
# Each element in x and y is (SubGroupKey, iterableResults)
# in which iterableResults are (PixelKey, features/labels)
x, y = get_image_rdd(sc, n_groups=1000, start=0, end=1300)

# knn model
knn = KNN(x,y)
del x, y

# test data
x_, y_ = get_image_rdd(sc, val=1)
x_list, y_list = x_.collect(), y_.collect()
del x_, y_
print 'Length of x_list', len(x_list)
cm = numpy.matlib.zeros((2,2), dtype=float)
pred = []
# iterate 10 pixels at a time
flag = True
counter = 0
while flag:
    x__, y__ = x_list[:10], y_list[:10]
    x_list[:10] = []
    y_list[:10] = []
    x__ = sc.parallelize(x__)
    y__ = sc.parallelize(y__)

    pred__, cm__ = knn.test(x__, y__, range(counter, counter+10))
    counter += 10

    pred.extend(pred__)
    cm = cm + cm__
    print 'Print cm', cm
    if len(x_list) < 1:
        flag = False
    del pred__, cm__, x__, y__

print cm
np.savetxt('./conf_mat.txt', cm)

pred_ = np.asarray([item[1] for item in pred])
np.savetxt('./predictions.txt', pred_)


sc.stop()

'''
x_ = np.array([[ 42,  60,  93,  38,  56,  88,  37,  55,  86,  42,  60,  92,  45,\
64,  95,  45,  63,  95,  44,  64,  97,  43,  64,  97,  52,  60,\
85, 199, 160, 152, 206, 164, 160, 182, 126, 122,  43,  64,  96,\
44,  64,  97,  45,  65,  98,  44,  66,  98,  45,  65,  95,  40,\
61,  93,  92,  74,  84, 188, 124, 122, 207, 146, 138, 212, 155,\
144, 153, 106, 103, 161, 112, 110,  42,  61,  93,  41,  60,  93,\
39,  60,  93,  35,  59,  93,  40,  64,  97,  75,  70,  83, 212,\
167, 156, 218, 173, 168,  41,  61,  93,  42,  61,  92,  70,  67,\
83,  97,  77,  83, 208, 147, 140, 238, 187, 183, 243, 198, 192,\
205, 151, 146,  38,  58,  91,  56,  60,  83,  85,  74,  83, 239,\
191, 180,  85,  74,  87, 190, 137, 131, 254, 210, 204, 254, 219,\
211, 196, 150, 143]], dtype=np.uint8)

y_ = np.array([[1]])
d = np.hstack((x_,y_))
d = sc.parallelize(d).zipWithIndex()
print d.collect()

x_ = d.map(lambda (d_,i_):(i_,d_[:-1]))
y_ = d.map(lambda (d_,i_):(i_,d_[-1]))
print x_.collect()
print y_.collect()

'''

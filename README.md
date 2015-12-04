# kNN4Spark
#### PySpark Implementation for k Nearest-Neighbor Classification

Test the code: `spark-submit knn.py`

##### Updates on 12/03/15:

I tried to use flatMap to directly input images and output pixels. It takes 40s to index and 40s to count for 100 images.
The relationship seems linear: 500 images cost 200s to index and 200s to count.

##### To do List (12/03/15):

* Sampling?

* Is it appropriate to do resize?

* Output confusion matrix in text format onto disk.

#### Strategies employed so far:

* Changing d -- pixel block size
* Changing N -- resizing
* Parallelization -- Spark/HPC
* Algebra -- weighting option (I think this would count) -- thoughts?? -> I think this is reasonable (Zhiya Zuo).
* What about changing distance metric? Do you think changing this will be considered as a strategy?
 
###### In training folder:
* Largest - img01114 (1280 x 960)
* Smallest - img00985 (99 x 117)



# kNN4Spark
#### PySpark Implementation for k Nearest-Neighbor Classification

Test the code: `spark-submit knn.py`

##### To do List (12/02/15):

* Testing training code -- More than 5,300 samples will give me outofmemory errors.

* Do you think we should also store the wrong pixels so that we can go back to see the "hard samples"? -- this will be helpful when writing the report. Will add more substance to the report.

* Output confusion matrix in text format onto disk.

#### Strategies employed so far:

* Changing d -- pixel block size.
* Parallelization -- HDFS/Spark/HPC
* Algebra -- weighting option (I think this would count) -- thoughts?? -> I think this is reasonable (Zhiya Zuo).
* What about changing distance metric? Do you think changing this will be considered as a strategy?



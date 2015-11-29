# kNN4Spark
#### PySpark Implementation for k Nearest-Neighbor Classification

Test the code: `spark-submit knn.py`

[knn_copy.py](https://github.com/zhiyzuo/kNN4Spark/blob/master/knn_copy.py) is just a copy of previous version (function). Since I rewrote the algorithm into a `class`, I saved the original file as a copy in case functions are better.

##### To do List:

* Testing training code -- done. worked well for weighted=False but gave ZeroDivisionError for weighted=True. Modified utils.py and no longer receiving ZeroDivisionError for weighted=True -- Updated on 11/28 920pm: I used offset instead of if-else syntax to avoid ZeroDivisionError.
* Do you think we should also store the wrong pixels so that we can go back to see the "hard samples"? -- this will be helpful when writing the report. Will add more substance to the report.
* Instance Function test()

#### Strategies employed so far:

* Changing d -- pixel block size.
* Parallelization -- HDFS/Spark/HPC
* Algebra -- weighting option (I think this would count) -- thoughts??



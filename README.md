# kNN4Spark
#### PySpark Implementation for k Nearest-Neighbor Classification

Test the code: `spark-submit knn.py`

##### Updates on 12/03/15:

I tried to use flatMap to directly input images and output pixels. It takes 40s to index and 40s to count for 100 images.
The relationship seems linear: 500 images cost 200s to index and 200s to count.

From Joel: Dataset Description (Without resizing)
- 5x5 block samples: 529,306,855
- 7x7 block samples: 525,249,341
- 9x9 block samples: 521,207,827
- Total number of skin pixels: 16.77%

##### To do List (12/05/15):

* Separate data into features and labels, with the same indices.

* To reduce the number of samples, we do resizing on images. Predictions will be reshaped to the original size and save as images using [`PIL.Image.fromarray(obj, mode=None)`](http://pillow.readthedocs.org/en/3.0.x/reference/Image.html).

* For each image, for its every pixel `p` (as RDD object), do `p.cartesian(train_rdd)` where `train_rdd` is reorgainzed to a `n1 x n2` matrix from a `1 x n` vector where `n1 x n2 == n`.

* Output confusion matrix in text format along with the binary image onto disk.

#### Strategies employed so far:

* Changing d -- pixel block size
* Changing N -- resizing
* Parallelization -- Spark/HPC
* Algebra -- weighting option (I think this would count) -- thoughts?? -> I think this is reasonable (Zhiya Zuo).
* What about changing distance metric? Do you think changing this will be considered as a strategy?
 
###### In training folder:
* Largest - img01114 (1280 x 960)
* Smallest - img00985 (99 x 117)



import numpy as np

img = np.loadtxt("/home/mvijayen/BDA/test.txt")
index = np.array(range(len(img)))
index = index.reshape(len(index),1)
features = img[:,0:-1]
cls=img[:,[-1]]
indClassFeat = np.concatenate((index,cls,features),axis=1)
distance(indClassFeat)

def cartesian (array1):
    from sklearn.utils.extmath import cartesian
    pairs = cartesian(array1, array1)
    return(pairs)
    
def distance (array1,norm=2):
    a1 = array1[0]
    a2 = array1[1]
    f1 = a1[:,2:]
    f2 = a2[:,2:]
    i1 = a1[:,[0]]
    i2 = a2[:,[0]]
    c1 = a1[:,[1]]
    c2 = a2[:,[1]]
    dist = np.linalg.norm(np.asarray(f1) - np.asarray(f2), norm)
    return(i1,i2,c1,c2,dist)

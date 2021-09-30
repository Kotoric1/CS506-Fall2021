import numpy as np
import scipy.spatial.distance as dist
def euclidean_dist(x, y):
    res = 0
    for i in range(len(x)):
        res += (x[i] - y[i])**2
    return res**(1/2)

def manhattan_dist(x, y):
    distance = np.sum(np.abs(x - y))
    return distance

def jaccard_dist(x, y):
    matrix=np.array([x,y])
    distance=dist.pdist(matrix,'jaccard')
    return distance

def cosine_sim(x, y):
    cosine=np.dot(x,y)/(np.linalg.norm(x)*(np.linalg.norm(y)))
    return cosine

# Feel free to add more

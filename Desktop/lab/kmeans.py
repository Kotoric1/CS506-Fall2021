from collections import defaultdict
from math import inf
import random
import csv
import numpy as np


def point_avg(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    (points can have more dimensions than 2)
    
    Returns a new point which is the center of all the points.
    """
    point=np.array(points)
    centroids=sum(point)/len(point)
    return centroids


def update_centers(dataset, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes 
    of both lists correspond to each other.
    Compute the center for each of the assigned groups.
    Return `k` centers in a list
    """
    raise NotImplementedError()

def assign_points(data_points, centers):
    """
    """
    assignments = []
    for point in data_points:
        shortest = inf  # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    """
    Returns the Euclidean distance between a and b
    """
    distance=np.sqrt(np.sum(np.power((a - b), 2)))
    return distance

def distance_squared(a, b):
    raise NotImplementedError()

def generate_k(dataset, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    """
    random_set=random.sample(dataset,k)
    return random_set

def cost_function(clustering):
    raise NotImplementedError()


def generate_k_pp(dataset, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    where points are picked with a probability proportional
    to their distance as per kmeans pp
    """
    m, n = np.shape(dataset)
    cluster_centers = np.mat(np.zeros((k, n)))
 
    index = np.random.randint(0, m)
    cluster_centers[0, ] = np.copy(dataset[index, ]) 

    d = [0.0 for _ in range(m)]
 
    for i in range(1, k):
        sum_all = 0
        for j in range(m):
            d[j] = nearest(dataset[j, ], cluster_centers[0:i, ])
            sum_all += d[j]
        sum_all *= random()
        for j, di in enumerate(d):
            sum_all -= di
            if sum_all > 0:
                continue
            cluster_centers[i] = np.copy(dataset[j, ])
            break
    return cluster_centers

def nearest(point, cluster_centers):
    m = np.shape(cluster_centers)[0]  
    for i in range(m):
        d = distance(point, cluster_centers[i, ])
        if min_dist > d:
            min_dist = d
    return min_dist


def _do_lloyds_algo(dataset, k_points):
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    clustering = defaultdict(list)
    for assignment, point in zip(assignments, dataset):
        clustering[assignment].append(point)
    return clustering


def k_means(dataset, k):
    if k not in range(1, len(dataset)+1):
        raise ValueError("lengths must be in [1, len(dataset)]")
    
    k_points = generate_k(dataset, k)
    return _do_lloyds_algo(dataset, k_points)


def k_means_pp(dataset, k):
    if k not in range(1, len(dataset)+1):
        raise ValueError("lengths must be in [1, len(dataset)]")

    k_points = generate_k_pp(dataset, k)
    return _do_lloyds_algo(dataset, k_points)

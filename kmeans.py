
import sys
import pandas as pd
import random as rand
import math

def diskKMeans(data, k):
    centroids = selectInitialCentroids(data, k)
    while(not done(centroids, clusters)):
        # each index in clusters represents the number of its centroid
        # so clusters[i] contains all the points that are attatched to the ith centroid
        clusters = [None * k]
        # arrange all the points into clusters
        for i in range(len(data)):
            cl_index = closest_cluster_index(data.iloc[i], centroids)
            clusters[cl_index] = data.iloc[i]
        # TODO adjust centroids according to mean of clusters


def closest_centroid_index(data_point, centroids):
    min_dist = math.inf
    min_centroid = None
    min_index = -1
    for i in range(len(centroids)):
        dist = get_dist(data_point, centroids[i])
        if(dist < min_dist):
            min_dist = dist
            min_centroid = centroids[i]
            min_index = i
    return min_index


def get_dist(test_point, train_point):
    distance = 0
    euc_distance = 0
    for i in range(len(train_point) - 1):
        if(type(train_point[i]) is str):
            if(test_point[i] != train_point[i]):
                distance += 1
        else:
            euc_distance += pow((test_point[x] - train_point[x]), 2)
    distance += math.sqrt(euc_distance)

    return distance


# stoppage conditions
def done(centroids, k):
    return False

# making this random for now
def selectInitialCentroids(data, k):
    centroids = []
    for i in range(k):
        centroids.append(rand.choice(data.values.tolist()))
    return centroids


def main():
    filename = None
    if(len(sys.argv) < 2):
        print("Usage: python kmeans.py <filename>")
        exit()
    else:
        filename = sys.argv[1]
    data = pd.read_csv(filename)
    k = 3
    print(diskKMeans(data, k))

if(__name__ == "__main__"):
    main()

import sys
import pandas as pd
import random as rand
import numpy as np
import math


class KMeans:
    def __init__ (self):
        pass

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

# euclidean distance of two or more points(array)
def calc_distance(p1, p2):
    x = np.sqrt(np.sum(np.power(p1-p2, 2), axis=1))
    return x

# stoppage condition 3
# sums all point distances to their respective centroid
# check if working
# prototype
def stoppage_sse(self, clusters, centroids, threshhold):
    sse = 0
    for i, c in enumerate(clusters):
        for p in c:
            sse += calc_distance(p, centroids[i])
    return abs(sse - self.prev_sse) < threshhold

# stoppage condition 2
# prototype
def centroids_changed(self, centroids, threshold):
    changed = False
    for i,c in enumerate(centroids):
        if abs(np.sum(c) - np.sum(self.prev_centroids[i])) > threshold:
            changed = True
    return changed
# stoppage condition 1
# protoype
def check_reasignments(self, clusters, threshold):
    changed = False
    for i, cl in enumerate(clusters):
        if abs(len(cl) - self.prev_len[i]) > threshold:
            changed = True
    return changed

def done(self, clusters, centroids, centroid_th, cluster_th, sse_th):
    return self.check_reasignments(clusters, cluster_th) or self.centroids_changed(centroids, centroid_th) or stoppage_sse(clusters, centroids, sse_th)

#find the centroid of the complete dataset
def find_dataset_centroid(data):
    return np.sum(data)/len(data)

# check if this makes sense
# should work
def selectInitialCentroids(data, k):
    centroids = []
    centroid = find_dataset_centroid(data)
    centroid_distances = calc_distance(data, centroid)
    for i in range(k):
        index_centroid = np.argmax(centroid_distances)
        centroid = data.iloc[index_centroid]
        centroids.append(centroid)
        centroid_distances += calc_distance(data, centroid)
    return centroids


def main():
    filename = None
    if(len(sys.argv) < 2):
        print("Usage: python kmeans.py <filename>")
        exit()
    else:
        filename = sys.argv[1]
    data = pd.read_csv(filename)
    data_wo_cl = data.drop(data.columns[4], axis = 1)
    k = 3
    print(diskKMeans(data, k))

if(__name__ == "__main__"):
    main()

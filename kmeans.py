
import sys
import pandas as pd
import random as rand
import math

def diskKMeans(data, k):
    centroids = selectInitialCentroids(data, k)
    print(centroids)
    attr_dict = get_attr_dict(data)
    while(not done(centroids, clusters)):
            # each index in clusters represents the number of its centroid
            # so clusters[i] contains all the points that are attatched to the ith centroid
        clusters = [None for i in range(k)]
        # arrange all the points into clusters
        for i in range(len(data)):
            cl_index = closest_cluster_index(data.iloc[i], centroids)
            clusters[cl_index] = data.iloc[i]
        for i in range(k):
            centroids[i] = average_point(clusters[i], attr_dict)

        print(centroids)




def average_point(cluster, attr_dict):
    average_point = []
    for data_point in cluster:
        # TODO data_point is float?
        for attr in data_point:
            if(str(attr).isdigit()):
                attr_dict[attr] += attr_dict[attr]
            else:
                attr_dict[attr].append(attr)

    i = 0
    for k,v in attr_dict:
        if(type(v) == list):
            average_point[i] = max(set(v), key=v.count)
        else:
            average_point[i] = v / len(cluster)
        i += 1

    return pd.Series(data=average_point)


def get_attr_dict(data):
    attr_dict = {}
    attrs = list(data.columns[:-1])

    for attr in attrs:
        if(str(attr).isdigit()):
            attr_dict[attr] = 0
        else:
            attr_dict[attr] = []

    return attr_dict



def closest_cluster_index(data_point, centroids):
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
            euc_distance += pow((test_point[i] - train_point[i]), 2)
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

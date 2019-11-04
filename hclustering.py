import pandas as pd
import sys
from scipy.spatial.distance import cdist
import numpy as np
import json
import matplotlib.pyplot as plt
import math


def hclustering(data, thresh, dist_type):
    data = data.iloc[:, :-1]
    clusters = to_cluster_list(data)
    while(len(clusters) > 1):
        dist_mx = calc_distance_matrix(clusters, dist_type)
        cluster1, cluster2, dist = find_closest_clusters(dist_mx, clusters)
        cluster = Cluster()
        cluster.left = cluster1
        cluster.right = cluster2
        cluster.dist = dist
        clusters.remove(cluster1)
        clusters.remove(cluster2)
        clusters.append(cluster)
        if(None in clusters):
            clusters.remove(None)
        print(len(clusters))

    output_to_json(clusters[0])

    if(thresh is not None):
        output_clusters(clusters, thresh)


def plot_clusters(clusters):

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    print(len(clusters))
    points = []
    for i in range(len(clusters)):
        points = clusters[i].get_all_points()
        for j in range(len(points)):
            plt.scatter(points[j].values[0], points[j].values[1], color=colors[i])

    plt.show()


def output_clusters(clusters, thresh):
    while(not done(clusters, thresh)):
        temp = []
        clusters_to_remove = []
        for i in range(len(clusters)):
            if(float(clusters[i].dist) > float(thresh)):
                temp.append(clusters[i].left)
                temp.append(clusters[i].right)
                clusters_to_remove.append(clusters[i])
        for i in range(len(clusters_to_remove)):
            clusters.remove(clusters_to_remove[i])
        clusters.extend(temp)

    plot_clusters(clusters)


def done(clusters, thresh):
    for i in range(len(clusters)):
        if(float(clusters[i].dist) > float(thresh)):
            return False
    return True



def output_to_json(root):
    res = {}
    res["type"] = "root"
    res["height"] = root.dist
    res["nodes"] = get_nodes(root.left, root.right)

    output = json.dumps(res, indent = 2)

    open("output.json", "w").write(output)


def get_nodes(left, right):
    res = []
    if(left.is_leaf()):
        left_dict = {}
        left_dict["type"] = "leaf"
        left_dict["height"] = left.dist
        left_dict["data"] = list(left.data)
    else:
        left_dict = {}
        left_dict["type"] = "node"
        left_dict["height"] = left.dist
        left_dict["nodes"] = get_nodes(left.left, left.right)

    res.append(left_dict)

    if(right.is_leaf()):
        right_dict = {}
        right_dict["type"] = "leaf"
        right_dict["height"] = right.dist
        right_dict["data"] = list(right.data)
    else:
        right_dict = {}
        right_dict["type"] = "node"
        right_dict["height"] = right.dist
        right_dict["nodes"] = get_nodes(right.left, right.right)

    res.append(right_dict)

    return res




def to_cluster_list(data):
    res = []
    for i in range(len(data)):
        cluster = Cluster()
        cluster.data = data.iloc[i]
        res.append(cluster)
    return res

def calc_distance_matrix(clusters, dist_type):
    mx = np.zeros((len(clusters), len(clusters))).astype(float)
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            mx[i,j] = clusters[i].dist_to(clusters[j], dist_type)
    return mx


def find_closest_clusters(dist_mx, clusters):
    min_val = 0
    distances = np.sort(dist_mx.flatten())
    for i in range(len(distances)):
        if(distances[i] != 0):
            min_val = distances[i]
            break

    cl_indexes = np.argwhere(dist_mx == min_val)
    cl_index1 = cl_indexes[0][0]
    cl_index2 = cl_indexes[0][1]

    return (clusters[cl_index1], clusters[cl_index2], min_val)



# if its a leaf, left and right are None
class Cluster():
    def __init__(self):
        self.left = None
        self.right = None
        self.dist = 0
        self.data = None

    def is_leaf(self):
        return (self.right is None and self.left is None)

    def dist_to(self, cluster, dist_type):
        if(self.is_leaf() and cluster.is_leaf() and self is cluster):
            return 0

        elif(dist_type == "single_link"):
            cluster_points1 = self.get_all_points()
            cluster_points2 = cluster.get_all_points()
            dist_mx = cdist(cluster_points1, cluster_points2, "euclidean")
            dist_mx = np.array(dist_mx)
            distances = np.sort(dist_mx.flatten())
            for i in range(len(distances)):
                if(distances[i] != 0):
                    return distances[i]
            return 0

        elif(dist_type == "complete_link"):
            cluster_points1 = self.get_all_points()
            cluster_points2 = cluster.get_all_points()
            dist_mx = cdist(cluster_points1, cluster_points2, "euclidean")
            dist_mx = np.array(dist_mx)
            return dist_mx.max()

        elif(dist_type == "average_link"):
            cluster_points1 = self.get_all_points()
            cluster_points2 = cluster.get_all_points()
            average1 = np.average(cluster_points1, axis=0)
            average2 = np.average(cluster_points2, axis=0)
            euc_distance = 0
            for i in range(len(average1)):
                euc_distance += pow((average1[i] - average2[i]), 2)
            return math.sqrt(euc_distance)
        else:
            raise NotImplementedError


    def get_all_points(self):
        res = []
        if(self.is_leaf()):
            res.append(self.data)
        else:
            res = self.right.get_all_points()
            res.extend(self.left.get_all_points())

        return res


    def __str__(self):
        return "Cluster(\n    " + str(self.left) + ", " + str(self.right) + "\n)"


def main():
    filename = None
    thresh = None
    dist_type = None
    if(len(sys.argv) < 3):
        print("Usage: python hclustering.py filename dist_type <threshold>")
        exit()
    elif(len(sys.argv) > 3):
        thresh = sys.argv[3]
        dist_type = str(sys.argv[2]).strip()
        filename = sys.argv[1]
    else:
        filename = sys.argv[1]
        dist_type = str(sys.argv[2]).strip()
    data = pd.read_csv(filename)

    hclustering(data, thresh, dist_type)



if(__name__ == "__main__"):
    main()

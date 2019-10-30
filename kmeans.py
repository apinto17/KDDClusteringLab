import sys
import pandas as pd
import random as rand
import numpy as np
import math


class KMeans:
    def __init__ (self):
        pass

    def diskKMeans(self, data, k):
        attr_dict = self.get_attr_dict(data)
    # TODO re-organize so that is works with initial conditions
        data = data.iloc[:, :-1]
        centroids = self.selectInitialCentroids(data, k)
        clusters = [[] for i in range(k)]
        for i in range(len(data)):
            cl_index = self.closest_cluster_index(data.iloc[i], centroids)
            clusters[cl_index].append(data.iloc[i])
        while(not self.done(centroids, clusters)):
            self.prev_centroids = centroids
                # each index in clusters represents the number of its centroid
                # so clusters[i] contains all the points that are attatched to the ith centroid
            # arrange all the points into clusters
            for i in range(k):
                centroids[i] = self.average_point(clusters[i], attr_dict)

            clusters = [[] for i in range(k)]
            for i in range(len(data)):
                cl_index = self.closest_cluster_index(data.iloc[i], centroids)
                clusters[cl_index].append(data.iloc[i])

            print(centroids)



    def average_point(self, cluster):
        average_point = np.average([dp.to_numpy() for dp in cluster], axis=0)

        return pd.Series(data=average_point)


    def closest_cluster_index(self, data_point, centroids):
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


    def get_dist(self, test_point, train_point):
        distance = 0
        euc_distance = 0
        for i in range(len(train_point)):
            if(type(train_point[i]) is str):
                if(test_point[i] != train_point[i]):
                    distance += 1
            else:
                euc_distance += pow((test_point[i] - train_point[i]), 2)
        distance += math.sqrt(euc_distance)

        return distance

    # euclidean distance of two or more points(array)
    def calc_distance(self, p1, p2):
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
    def find_dataset_centroid(self, data):
        return np.sum(data)/len(data)

    # check if this makes sense
    # should work
    def selectInitialCentroids(self, data, k):
        centroids = []
        centroid = self.find_dataset_centroid(data)
        centroid_distances = self.calc_distance(data, centroid)
        index_centroid = np.argmax(centroid_distances)
        centroid = data.iloc[index_centroid]
        centroids.append(centroid)
        distances = self.calc_distance(data, centroid)
        for i in range(1,k):
            index_centroid = np.argmax(distances)
            centroid = data.iloc[index_centroid]
            centroids.append(centroid)
            distances += self.calc_distance(data, centroid)
        return centroids


def main():
    filename = None
    if(len(sys.argv) < 2):
        print("Usage: python kmeans.py <filename>")
        exit()
    else:
        filename = sys.argv[1]
    data = pd.read_csv(filename)

    kmeans = KMeans()
    k = 3
    print(kmeans.diskKMeans(data, k))

if(__name__ == "__main__"):
    main()

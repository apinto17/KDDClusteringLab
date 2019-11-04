import sys
import pandas as pd
import random as rand
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm

class KMeans:
    def __init__ (self, stoppage=2, threshold=1):
        self.stoppage = stoppage
        self.threshold = threshold
        self.prev_sse = 0

    def init_centroids(self, centroids):
        self.prev_centroids = [None] * len(centroids)
        for i, c in enumerate(centroids):
            self.prev_centroids[i] = c.copy(deep=True)
            self.prev_centroids[i].values[:] = 0

    def update_cluster_lengths(self, clusters):
        for i in range(len(clusters)):
            self.prev_len[i] = len(clusters[i])
    
    #hacky plot
    def plot_clusters(self, clusters, centroids):
        pca = PCA(n_components=2) #2-dimensional PCA
        my_color = ['b','r','g', 'y']
        for ci, cv in enumerate(clusters):
            for i in cv:
                plt.scatter(i.values[0], i.values[1], color=my_color[ci])
            plt.scatter(centroids[ci][0], centroids[ci][1], color=my_color[ci], s=3)
        plt.show()

    def plot_pca(self, clusters, centroids):
        pca = PCA(n_components=2) #2-dimensional PCA
        colors = cm.rainbow(np.linspace(0,1, len(clusters)))
#        colors = ['r', 'g', 'b']
        data = pd.DataFrame(clusters[0])
        data['cluster'] = np.zeros(len(clusters[0]))
        for c in range(1,len(clusters)):
            cluster_data = np.ones(len(clusters[c])) * c
            new_data = pd.DataFrame(clusters[c])
            new_data['cluster'] = cluster_data
            new_cols = {x: y for x, y in zip(data.columns, new_data.columns)}
            data = data.append(new_data.rename(columns=new_cols))

        x = data['cluster'].values.astype(int)
        data = data.drop('cluster', axis=1)
        data = StandardScaler().fit_transform(data)
        transformed = pd.DataFrame(pca.fit_transform(data), columns=['y', 'y2'])
        for ci, cv in enumerate(transformed.values):
            plt.scatter(cv[0], cv[1], color=colors[x[ci]])
        plt.show()
        


    def diskKMeans(self, data, k):
        centroids = self.selectInitialCentroids(data, k)
        self.init_centroids(centroids)
        clusters = [[] for i in range(k)]
        for i in range(len(data)):
            cl_index = self.closest_cluster_index(data.iloc[i], centroids)
            clusters[cl_index].append(data.iloc[i])
        
        self.prev_len = [0] * k
        while(not self.done(clusters, centroids)):
            self.prev_centroids = centroids.copy()
            self.update_cluster_lengths(clusters)
            for i in range(k):
                centroids[i] = self.average_point(clusters[i])

            clusters = [[] for i in range(k)]
            for i in range(len(data)):
                cl_index = self.closest_cluster_index(data.iloc[i], centroids)
                clusters[cl_index].append(data.iloc[i])
        
        self.clusters = clusters
        self.centroids = centroids
        return clusters, centroids
        

    def output_data(self):
        for i in range(len(self.clusters)):
            print("Cluster: "+str(i))
            print("Centroids: ", end='')
            print(self.centroids[i].values)
            cluster_points = pd.DataFrame(self.clusters[i])
            dist_vec = self.calc_distance(cluster_points.values, self.centroids[i].values)

            print("Max Dist. to Center: %f" % (np.max(dist_vec)))
            print("Min Dist. to Center: %f" % (np.min(dist_vec)))
            print("Avg Dist. to Center: %f" % (np.average(dist_vec)))
            print("%d Points:" % (len(self.clusters[i])))
            for p in cluster_points.values:
                print(np.array2string(p, precision=2, separator=','))
            print("End cluster %d" % i)






    def average_point(self, cluster):
        average_point = np.average([dp.to_numpy() for dp in cluster], axis=0)
        x = pd.Series(data=average_point)

        return x


    def closest_cluster_index(self, data_point, centroids):
        min_dist = math.inf
        min_centroid = None
        min_index = -1

        for i in range(len(centroids)):
            dist = self.get_dist(data_point, centroids[i])
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
    def calc_distance_single(self, p1, p2):
        x = np.sqrt(np.sum(np.power(p1.values-p2.values, 2)))
        return x

    # stoppage condition 3
    # sums all point distances to their respective centroid
    # check if working
    # prototype
    def stoppage_sse(self, clusters, centroids, threshhold):
        sse = 0
        for i, c in enumerate(clusters):
            for p in c:
                sse += self.calc_distance_single(p, centroids[i])
        done =  abs(sse - self.prev_sse) < threshhold
        self.prev_sse = sse
        return done

    # stoppage condition 2
    # prototype
    def centroids_changed(self, centroids, threshold):
        for i,c in enumerate(centroids):
            if abs(np.sum(c) - np.sum(self.prev_centroids[i])) > threshold:
                return False
        return True
    # stoppage condition 1
    # protoype
    def check_reasignments(self, clusters, threshold):
        for i, cl in enumerate(clusters):
            if abs(len(cl) - self.prev_len[i]) > threshold:
                return False
        return True

    def done(self, clusters, centroids):
        if self.stoppage == 0:
            return self.check_reasignments(clusters, self.threshold)
        elif self.stoppage ==1:
            return self.centroids_changed(centroids, self.threshold)
        else:
            return self.stoppage_sse(clusters, centroids, self.threshold)

    #find the centroid of the complete dataset
    def find_dataset_centroid(self, data):
        return np.sum(data)/len(data)

    # check if this makes sense
    # should work
    def selectInitialCentroids(self, data, k):
        centroids = []
        centroid = self.find_dataset_centroid(data)
        centroid_distances = self.calc_distance(data, centroid)
        index_centroid = np.argmax(np.array(centroid_distances.values))
        centroid = data.iloc[index_centroid]
        centroids.append(centroid)
        distances = self.calc_distance(data, centroid)
        for i in range(1,k):
            index_centroid = np.argmax(np.array(distances.values))
            centroid = data.iloc[index_centroid]
            centroids.append(centroid)
            distances += self.calc_distance(data, centroid)
        return centroids


def main():
    filename = None
    kmeans = None
    stoppage = 2
    if(len(sys.argv) < 3):
        print("Usage: python kmeans.py <filename> <k> [<stoppage> <threshold>]")
        exit()
    elif(len(sys.argv) == 5):
        kmeans = KMeans(int(sys.argv[3]), float(sys.argv[4]))
    else:
        kmeans = KMeans()
    
    filename = sys.argv[1]
    k = int(sys.argv[2])
    data = pd.read_csv(filename)
    data.drop(list(data.filter(regex = '0')), axis = 1, inplace = True)

    # comment out if you don't want 2d data
    #data = pd.DataFrame(np.random.randint(0,100,size=(100, 3)))

    clusters, centroids = kmeans.diskKMeans(data, k)
    # comment out if no plot wanted
    kmeans.output_data()
    kmeans.plot_pca(clusters, centroids)

if(__name__ == "__main__"):
    main()

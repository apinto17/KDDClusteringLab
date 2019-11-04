import sys
import pandas as pd
import random as rand
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

class Density:
    # eps = epsilon/radios 
    def __init__(self, eps, minPts):
        self.eps = eps
        self.minPts = minPts
    
    def calc_clusters(self, data, dist_method):
        m = len(data)
        point_type = np.zeros(m).astype(int)

#        dist_mx = self.calc_distance_matrix(data.values)
        dist_mx = cdist(data, data, dist_method)
        C = 0
        for i in range(m):
            if point_type[i] != 0:
                continue
            neighbours=np.where(dist_mx[i]<self.eps)[0]
            if len(neighbours) < self.minPts:
                point_type[i] = -1
                continue

            C += 1
            point_type[i] = C
            neighbours = neighbours[1:]
            while(len(neighbours) > 0):
                n = neighbours[0]
                if point_type[n] == -1:
                    point_type[n] = C
                if point_type[n] == 0:
                    point_type[n] = C
                    new_neighbours = np.where(dist_mx[n] < self.eps)[0]
                    if len(new_neighbours) >= self.minPts:
                        neighbours = np.append(neighbours, new_neighbours)
                neighbours = neighbours[1:]
        return point_type


#    def calc_distance_matrix(self, points):
#        mx = np.zeros((len(points), len(points))).astype(float)
#        for k, p1 in enumerate(points):
#            for m, p2 in enumerate(points):
#                mx[k,m] = self.dist_calc.calc_distance(p1,p2)
#        return mx

    def output_data(self, data, types):
        print("minPoints: %d" % self.minPts)
        print("Radius: %f" % self.eps)
        uniques = np.unique(types)
        uniques = np.delete(uniques, np.argwhere(uniques == -1))
        outliers = np.where(types == -1)[0]
        print("Outliers: %d" % (len(outliers)))
        percentage = len(outliers)/len(data)
        print("Outliers in percent: %f" % (percentage))
        for i,u in enumerate(uniques):
            print("Cluster: "+str(i))
            indicies = np.where(types == u)[0]
            print("%d Points:" % (len(indicies)))
            for p in indicies:
                print(np.array2string(data.values[p], precision=2, separator=','))
            
            print("End cluster %d" % i)
        
        print("Outliers:")
        for i in range(min(73, len(outliers))):
            print(np.array2string(data.values[outliers[i]], precision=2, separator=','))


    def plot_points(self, points, types):
        unique = np.unique(types)
        colors = cm.rainbow(np.linspace(0,1, len(unique)))
        for i, p in enumerate(points.values):
            if types[i] == -1:
                color = 'k'
            else:
                color = colors[types[i]-1]
            
            plt.scatter(p[0], p[1], color=color)
        
        plt.show()
    

    def plot_pca(self, points, types):
        unique = np.unique(types)
        pca = PCA(n_components=2) #2-dimensional PCA
        colors = cm.rainbow(np.linspace(0,1, len(unique)))
        data = StandardScaler().fit_transform(points)
        transformed = pd.DataFrame(pca.fit_transform(data), columns=['y', 'y2'])

        for i, p in enumerate(transformed.values):
            if types[i] == -1:
                color = 'k'
            else:
                color = colors[types[i]-1]
            
            plt.scatter(p[0], p[1], color=color)
        plt.show()


def main():
    filename = None
    if(len(sys.argv) < 5):
        print("Usage: python kmeans.py <filename> <epsilon> <numPoints> <distance[euclidean|chebyshev|cityblock]>")
        exit()
    
    density = Density(float(sys.argv[2]),int(sys.argv[3]))
    filename = sys.argv[1]
    distance = sys.argv[4]
    data = pd.read_csv(filename)
    data.drop(list(data.filter(regex = '0')), axis = 1, inplace = True)

    # comment out if you don't want 2d data
    #---------------------------------------------------
#    blobs = datasets.make_blobs(centers=4, n_samples=100, random_state=8)
#    data = pd.DataFrame(blobs[0])

#    circles = datasets.make_circles(n_samples=1000, factor=.5, noise=.05)
#    data = pd.DataFrame(circles[0])
    #--------------------------------------------------
    point_types = density.calc_clusters(data, distance)

    density.output_data(data, point_types)
    density.plot_pca(data,point_types)


if(__name__ == "__main__"):
    main()

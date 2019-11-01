import sys
import pandas as pd
import random as rand
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import cdist
from sklearn import datasets

class Distance(ABC):
    @abstractmethod
    def calc_distance(self, p1, p2):
        pass

class Euklidean_Distance(Distance):
    def calc_distance(self, p1, p2):
        x = np.sqrt(np.sum(np.power(p1-p2, 2)))
        return x


class Density:
    def __init__(self, eps, minPts, distance_method):
        self.eps = eps
        self.minPts = minPts
        self.dist_calc = distance_method
    
    def calc_clusters(self, data):
        m = len(data)
#        labels = np.zeros(m)
        point_type = np.zeros(m).astype(int)

#        dist_mx = self.calc_distance_matrix(data.values)
        dist_mx = cdist(data, data, 'euclidean')
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


    def calc_distance_matrix(self, points):
        mx = np.zeros((len(points), len(points))).astype(float)
        for k, p1 in enumerate(points):
            for m, p2 in enumerate(points):
                mx[k,m] = self.dist_calc.calc_distance(p1,p2)
        return mx
    
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



def main():
    filename = None
    if(len(sys.argv) < 2):
        print("Usage: python kmeans.py <filename>")
        exit()
    else:
        filename = sys.argv[1]
    data = pd.read_csv(filename)
    data = data.iloc[:, :-1]

    # comment out if you don't want 2d data
    #---------------------------------------------------
    blobs = datasets.make_blobs(centers=5, n_samples=1500, random_state=8)
    data = pd.DataFrame(blobs[0])

    circles = datasets.make_circles(n_samples=1000, factor=.5, noise=.05)
    data = pd.DataFrame(circles[0])
    #--------------------------------------------------

    dc = Euklidean_Distance()

    density = Density(0.2,4, dc)
    point_types = density.calc_clusters(data)

    density.plot_points(data,point_types)


if(__name__ == "__main__"):
    main()

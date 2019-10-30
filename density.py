import sys
import pandas as pd
import random as rand
import numpy as np
from abc import ABC, abstractmethod

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
        dist_mx = self.calc_distance_matrix(data.values)


    def calc_distance_matrix(self, points):
        mx = np.zeros((len(points), len(points))).astype(float)
        for k, p1 in enumerate(points):
            for m, p2 in enumerate(points):
                mx[k,m] = self.dist_calc.calc_distance(p1,p2)
        return mx

def main():
    filename = None
    if(len(sys.argv) < 2):
        print("Usage: python kmeans.py <filename>")
        exit()
    else:
        filename = sys.argv[1]
    data = pd.read_csv(filename)
    data = data.iloc[:, :-1]

    dc = Euklidean_Distance()

    density = Density(2,3, dc)
    density.calc_clusters(data)


if(__name__ == "__main__"):
    main()

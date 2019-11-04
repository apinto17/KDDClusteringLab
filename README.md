## Authors
Alex Pinto, Sebastian Heusinger
## needed libraries
* numpy
* pandas
* matplotlib
* sklearn
* scipy
## k-means Clustering:
* &lt;stoppage&gt; selects the stoppage condition used. 0 == reasignment; 1 == cluster_centroids; 2 == sse_threshold
* &lt;threshold&gt; is the threshold for the selected algorithm

if you dont use them the default is sse_threshold and a threshold of 1
```bash
python kmeans.py <filepath> <k> [<stoppage> <threshold>]
```

## Hierarchical Clustering:

## DBSCAN
```bash
python dbscan.py <filepath> <epsilon> <NumPoints> <distance[euclidean|chebyshev|cityblock]>
```
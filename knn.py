# Implementation of KNN, used a package for efficiency with the emails DSET
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import mode
def knn_point(test_dpoint, train_points, k, ret_dist=False):
    dims = train_points.shape[1] -1
    test_dpoint = np.reshape(test_dpoint, (1, dims+1))
    distances = cdist(train_points[:, :dims], test_dpoint[:, :dims])
    knn_idx = np.argsort(distances, axis=0)[:k].reshape(k)
    knn = train_points[knn_idx,:]
    classification = mode(knn[:, dims]).mode[0]
    if ret_dist:
        return classification, distances
    return classification

def knn(training_data, test_data, k):
    final_arr = np.apply_along_axis(knn_point, axis=1, arr=test_data, train_points=training_data, k=k)
    return final_arr
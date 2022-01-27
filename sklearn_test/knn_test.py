"""
Author: ANTenna on 2021/12/26 9:58 下午
aliuyaohua@gmail.com

Description:

demo of:
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.kneighbors
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


if __name__ == '__main__':
    # the data to be index
    samples = np.array([[0, 0],
                        [1, 1],
                        [2, 2]])

    neigh = NearestNeighbors(n_neighbors=1)  # 1-nn
    neigh.fit(samples)

    model = np.array([[0.1, 0.1],
                      [2.1, 2.1]])
    dist, idx = neigh.kneighbors(model)

    print('dist: \n', dist)
    print('idx: \n', idx.flatten())

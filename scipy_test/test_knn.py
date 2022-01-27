"""
Author: ANTenna on 2021/12/18 9:09 下午
aliuyaohua@gmail.com

Description:

"""

from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    X = np.array([[-1, -1],
                  [-2, -1],
                  [-3, -2],
                  [1, 1],
                  [2, 1],
                  [3, 2]])
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    print(indices)

    # vis
    for i, x in enumerate(X):
        plt.scatter(x[0], x[1])
        plt.text(x[0], x[1], i)

    plt.show()

    print()

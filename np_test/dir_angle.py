import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import linalg
from sklearn.decomposition import PCA

x = np.arange(1, 17, 1)
y = np.array([4.00, 6.40, 8.00, 8.80, 9.22,
              9.50, 9.70, 9.86, 10.00, 10.20, 10.32,
              10.42, 10.50, 10.55, 10.58, 10.60])
location = np.array([x, y]).T  # .reshape((-1, 2))

print(location)

sample_num = 6

start_traj = location[:sample_num]
end_traj = location[-sample_num:]
# print('end_traj is ', end_traj)
# end_traj = np.array([[5, 5], [6, 5.1], [7, 5.1], [8, 5],
#                      [10, 5], [11, 5], [12, 5], [15, 5.5]])

# get principle vector
s_move_vec = start_traj[-1] - start_traj[0]  # local_move_vec
e_move_vec = end_traj[-1] - end_traj[0]  # local_move_vec

ax = plt.axes()

pca_s = PCA(n_components=2)
pca_s.fit(start_traj)

pca_e = PCA(n_components=2)
pca_e.fit(end_traj)
# print('explained_variance_ratio_ ', pca_e.explained_variance_ratio_)
# print(pca.explained_variance_)

s = pca_s.components_[0]
e = pca_e.components_[0]

# real dir in time line
if np.dot(s_move_vec, s) < 0:
    s = -s
if np.dot(e_move_vec, e) < 0:
    e = -e

# start_angle = np.arccos(s_vh[0])
# end_angle = np.arccos(e_vh[0])
# print('start_angle is ', start_angle)
# print('end_angle is ', end_angle)

print('pca_s ', pca_s.components_)

plt.plot(location[:, 0], location[:, 1], '*')
plt.plot(start_traj[:, 0], start_traj[:, 1], 'o')
plt.plot(end_traj[:, 0], end_traj[:, 1], 'o')

ax.arrow(start_traj[0][0], start_traj[0][1],
         s[0]*20, s[1]*20,
         head_width=3, head_length=3
         )

# e_vh = e_vh
ax.arrow(end_traj[0][0], end_traj[0][1],
         e[0]*20, e[1]*20,
         head_width=3, head_length=3
         )

plt.show()

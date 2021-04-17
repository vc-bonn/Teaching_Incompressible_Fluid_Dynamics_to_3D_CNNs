import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D
fig = plt.figure()

ax = Axes3D(fig)

#x = np.load("imgs/voxel_grid_Wing.npy")
x = np.load("imgs/voxel_grid_3_objects.npy")
#x = np.load("imgs/voxel_grid_Submarine.npy")

print(f"x.shape: {x.shape}")

ax.voxels(x)

plt.show()

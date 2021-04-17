"""
Voxelization script
tested in Blender 2.83
important note: make sure, that object has closed surface
                and that surface normals are set up correctly!
"""

import bpy
from mathutils import Vector
import numpy as np

def is_inside(obj,p):
    """
    :obj: Object to check (e.g. bpy.data.objects["Cube"])
    :p: Vector of point to check (e.g. Vector((1,2,3)))
    :return: True if point is inside object, False otherwise
    """
    _,point,normal,face = obj.closest_point_on_mesh(p)
    p2 = point - p
    v = p2.dot(normal)
    return (v>0.0)

def voxelize(obj_name,ranges):
    """
    :obj_name: name of Object to voxelize
    :ranges: list of ranges for x,y,z (e.g. [np.arange(-1,1,0.1) for _ in range(3)])
    :return: 3d numpy array containing voxel information (1 for inside / 0 for outside)
    """
    obj = bpy.data.objects[obj_name]
    voxel_grid = np.zeros([len(ranges[0]),len(ranges[1]),len(ranges[2])])
    for i,x in enumerate(ranges[0]):
        for j,y in enumerate(ranges[1]):
            for k,z in enumerate(ranges[2]):
                if is_inside(obj,Vector((x,y,z))):
                    voxel_grid[i,j,k] = 1
    return voxel_grid

obj_name = "3_objects"#"Cyber"#"Submarine"#"Cube"#
scale = 0.2
x_range = np.arange(-6,1.51,scale)
y_range = np.arange(-4,4.01,scale)
z_range = np.arange(-1.5,1.51,scale)

print(f"start voxelization of {obj_name}...")

voxel_grid = voxelize(obj_name,[x_range,y_range,z_range])

print(f"saving to numpy file ({voxel_grid.shape})...")

np.save(f"voxel_grids/voxel_grid_{obj_name}.npy",voxel_grid)

print("done.")
import os
import glob
import numpy as np
import open3d as o3d


def show_pointcloud(pcd_path, is_mesh=1):
    print("Load a ply point cloud, print it, and render it")
    if is_mesh:
        pcd = o3d.io.read_triangle_mesh(pcd_path)
    else:
        pcd = o3d.io.read_point_cloud(pcd_path)
    print('pcd loaded ... ')

    o3d.visualization.draw_geometries(pcd, window_name='ANTenna3D')


if __name__ == '__main__':
    root_path = '/Users/aibee/PycharmProjects/pythonProject/test/open3d_test'
    plane_root_path = os.path.join(root_path, 'pcd_file', 'fine_grain_plane')

    # cropped
    # fine grain plane
    # plane_name = '1F-crop-down.ply'
    plane_file_list = sorted(glob.glob(plane_root_path + '/plane_patch*'))

    plane_name = 'plane_patch_1num_1188942.ply'
    pcd_path = os.path.join(root_path, 'pcd_file', 'fine_grain_plane', plane_name)

    # load batch plane
    pcd_list = []
    for p in plane_file_list:
        pcd_tmp = o3d.io.read_point_cloud(p)
        pcd_list.append(pcd_tmp)

    # hand select
    # pcd_path_list = ['plane_patch_1num_274452.ply', 'plane_patch_2num_213184.ply',
    #                  'plane_patch_3num_141367.ply', 'plane_patch_4num_111739.ply']
    # show_pointcloud(pcd_path=pcd_path, is_mesh=0)
    # for i in range(len(pcd_path_list)):
    #     # pcd_list[i] = plane_root_path + pcd_list[i]
    #     pcd_tmp = o3d.io.read_point_cloud(os.path.join(plane_root_path, pcd_path_list[i]))
    #     pcd_list.append(pcd_tmp)

    o3d.visualization.draw_geometries(pcd_list, window_name='ANTenna3D')

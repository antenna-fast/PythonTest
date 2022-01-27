"""
Author: ANTenna on 2021/12/29 9:26 下午
aliuyaohua@gmail.com

Description:
fine grain plane segmentation for general depth-render solution
"""

import os
import sys
import numpy as np
import open3d as o3d


if __name__ == '__main__':
    """
    1. load point cloud
    2. voxel filter to de-noise
    3. extract major plane from whole point cloud
    4. delete inlier point from the whole points
    5. iterative extract remainder fine-grain plane from remainder  
    """

    # 0. parameter
    is_downsample = 0

    # 1. io
    root_path = '/Users/aibee/PycharmProjects/pythonProject/test/open3d_test/pcd_file'
    # root_path = '/data0/texture_data/yaohualiu/project_info/UNIVERSAL/beijing/prod/1F'
    # pcd_path = os.path.join(root_path, '1F-crop.ply')
    pcd_path = os.path.join(root_path, 'fine_grain_plane', '1F-crop-down.ply')
    down_pcd_path = os.path.join(root_path, 'fine_grain_plane', '1F-crop-down.ply')

    # log
    f = open(os.path.join(root_path, 'fine_grain_plane', 'iterative_plane_segmentation.log'), 'w')

    wstring = 'loading pcd: {}... \n'.format(pcd_path)
    print(wstring)
    f.write(wstring)
    scene_pcd = o3d.io.read_point_cloud(pcd_path)

    # 2. voxel filter
    if is_downsample:
        print('down sampling ... ')
        f.write('down sampling pcd ... \n')
        voxel_size = 0.1
        scene_pcd = scene_pcd.voxel_down_sample(voxel_size)

        # estimate normal
        wstring = 'estimating normal ...'
        print(wstring)
        f.write(wstring)
        scene_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=30))

        # save down sampled pcd
        o3d.io.write_point_cloud(down_pcd_path, scene_pcd)

    # 3-4
    # parameters
    num_remainder_points_threshold = 100000
    num_plane_inlier = 80000  # we do not care about it if #inlier < set num

    dist_threshold = 0.025
    num_sample = 3
    num_iter = 1000

    f.write('parameter: \n')
    # f.write('down sample voxel size = {} \n'.format(voxel_size))
    f.write('num_remainder_threshold={} \n'.format(num_remainder_points_threshold))
    f.write('num_plane_inlier={}\n'.format(num_plane_inlier))

    plane_patch = 0

    # if 0:
    while len(np.array(scene_pcd.points)) > num_remainder_points_threshold:
        plane_patch += 1
        wstring = 'current iter: {} \n'.format(plane_patch)
        print(wstring)
        f.write(wstring)

        plane_model, inliers = scene_pcd.segment_plane(distance_threshold=dist_threshold,
                                                       ransac_n=num_sample,
                                                       num_iterations=num_iter)
        num_inlier = len(inliers)
        if num_inlier < num_plane_inlier:
            wstring = 'all plane segmented ... \n'
            print(wstring)
            f.write(wstring)
            break
        wstring = 'plane detected: num_inlier={} \n'.format(len(inliers))
        print(wstring)
        f.write(wstring)

        # plane parameter
        [a, b, c, d] = plane_model
        wstring = "Plane equation: {0:.2f}x + {1:.2f}y + {2:.2f}z + {3:.2f} = 0 \n".format(a, b, c, d)
        print(wstring)
        f.write(wstring)

        # inlier
        inlier_cloud = scene_pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])  # inlier is in red
        # outlier
        outlier_cloud = scene_pcd.select_by_index(inliers, invert=True)

        # 可视化
        # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
        plane_patch_file = os.path.join(root_path, 'fine_grain_plane', 'plane_patch_' + str(plane_patch) + 'num_' + str(num_inlier) + '.ply')
        o3d.io.write_point_cloud(plane_patch_file, inlier_cloud)

        # outlier_path = o3d.path.join(root_path, 'fine_grain_plane', 'outlier_' + str(plane_patch) + 'num_' + str(num_inlier) + '.ply')
        # o3d.io.write_point_cloud(outlier_path, outlier_cloud)

        # update scene by outlier
        scene_pcd = outlier_cloud

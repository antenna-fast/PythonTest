import os
import numpy as np
import open3d as o3d


def show_pointcloud(pcd_path, is_mesh=1):
    print("Load a ply point cloud, print it, and render it ... ")
    if is_mesh:
        pcd = o3d.io.read_triangle_mesh(pcd_path)
    else:
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # pcd.paint_uniform_color([0, 0.40, 0.1])
    # pcd_np = np.asarray(pcd.points)  # 将点转换为numpy数组

    o3d.visualization.draw_geometries([pcd],
                                      window_name='ANTenna3D',
                                      )


if __name__ == '__main__':
    # single frame
    frameIdx = 0
    pcd_path = '/Users/aibee/Downloads/File/RemoteData/PointCloudFrame/frame_{}.ply'.format(frameIdx)

    show_pointcloud(pcd_path=pcd_path, is_mesh=0)

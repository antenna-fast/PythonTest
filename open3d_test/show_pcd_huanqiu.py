import os
import numpy as np
import open3d as o3d


def show_pointcloud(pcd_path, is_mesh=1):
    print("Load a ply point cloud, print it, and render it")
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
    root_path = '/Users/aibee/PycharmProjects/pythonProject/test/open3d_test'

    # rgbd ch
    # ch = 'ch05008'
    # ch = 'ch05002'

    # cropped
    # ch = 'ch03010'
    # ch = 'ch03014'
    # ch = 'ch03016'
    ch = 'ch04002'
    # ch = 'ch04008'

    # pcd_path = os.path.join(root_path, 'pcd_file/ground_plane.ply')
    # pcd_path = os.path.join(root_path, 'pcd_file/1F-crop.ply')
    # pcd_path = os.path.join(root_path, 'pcd_file/1F-crop_mesh.ply')
    # pcd_path = os.path.join(root_path, 'pcd_file/ground_plane_mesh.ply')

    # pcd_path = os.path.join(root_path, 'pcd_file', ch, 'rgbd_point_cloud.ply')
    pcd_path = os.path.join(root_path, 'pcd_file', ch, 'pcd_whole.ply')

    show_pointcloud(pcd_path=pcd_path, is_mesh=0)

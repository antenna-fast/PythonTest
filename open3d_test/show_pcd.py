import os
import open3d as o3d


def show_pointcloud(pcd_path):
    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud(pcd_path)
    o3d.visualization.draw_geometries([pcd],
                                      window_name='ANTenna3D',
                                      # zoom=0.3412,
                                      # front=[0.4257, -0.2125, -0.8795],
                                      # lookat=[2.6172, 2.0475, 1.532],
                                      # up=[-0.0694, -0.9768, 0.2024]
                                      )


if __name__ == '__main__':
    ply_root = '/Users/aibee/Downloads/RemoteFile/SfmPly'
    # pcd_path = os.path.join(ply_root, 'output_points.ply')
    # pcd_path = os.path.join(ply_root, 'output_cameras.ply')
    pcd_path = os.path.join(ply_root, 'output_office_points.ply')
    # pcd_path = os.path.join(ply_root, 'output_office_cameras.ply')

    # colmap
    '/Users/aibee/Downloads/RemoteFile/colmap/0'
    show_pointcloud(pcd_path=pcd_path)

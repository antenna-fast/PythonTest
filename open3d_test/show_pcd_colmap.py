import os
import open3d as o3d


def show_pointcloud(pcd_path):
    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud(pcd_path)
    o3d.visualization.draw_geometries([pcd], window_name='ANTenna3D',)


if __name__ == '__main__':
    # colmap
    scene = 'scan9'
    ply_root = '/Users/aibee/Downloads/RemoteFile/colmap/' + scene + '/0'
    pcd_path = os.path.join(ply_root, 'fused.ply')
    show_pointcloud(pcd_path=pcd_path)

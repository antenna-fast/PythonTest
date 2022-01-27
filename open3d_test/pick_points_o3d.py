"""
Author: ANTenna on 2021/12/20 3:25 下午
aliuyaohua@gmail.com

Description:

"""

import open3d as o3d
import os


def pick_points(pcd):
    print("")
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


if __name__ == '__main__':
    # read pcd
    root_path = '/Users/aibee/Downloads/Dataset/3DDataset/6DPose/UWA/model'
    pcd = o3d.io.read_point_cloud(os.path.join(root_path, 'cheff.ply'))

    # paint color
    # pcd.paint_uniform_color([0.0, 0.5, 0.1])

    # build tree
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    picked_pt = pick_points(pcd)
    print('pick:', picked_pt)  # print selected point

    print()

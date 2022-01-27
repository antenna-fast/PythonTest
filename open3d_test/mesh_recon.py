"""
Author: ANTenna on 2021/12/20 6:21 下午
aliuyaohua@gmail.com

Description:
Poisson Reconstruction

"""

import os
import open3d as o3d


def show_pcd(pcd, window_name=''):
    o3d.visualization.draw_geometries([pcd],
                                      window_name='ANTenna3D' + '_' + window_name)


if __name__ == '__main__':
    # parameter
    is_remote = 1

    # ~.ply
    # pcd_file = 'ground_plane'
    # pcd_file = '1F'
    pcd_file = '1F-crop'

    is_vis = 0

    # recon methods
    recon_type = 'Poisson'
    # recon_type = 'Alpha'
    # recon_type = 'BallPivoting'

    # file path
    if is_remote:
        root_path = '/data0/texture_data/yaohualiu/project_info/UNIVERSAL/beijing/prod/1F'
    else:
        root_path = '/Users/aibee/PycharmProjects/pythonProject/test/open3d_test/pcd_file'

    # ground plane or whole pcd to mesh
    pcd_file_path = os.path.join(root_path, pcd_file + '.ply')
    output_mesh_path = os.path.join(root_path, pcd_file + '_mesh.ply')
    print('pcd_file_path: {}'.format(pcd_file_path))
    print('output_path: {}'.format(output_mesh_path))

    # read point cloud
    print('loading point cloud: {} ... '.format(pcd_file_path))
    pcd = o3d.io.read_point_cloud(pcd_file_path)

    if is_vis:
        show_pcd(pcd, window_name='point')

    print('down sampling ... ')
    voxel_size = 0.5
    print('voxel_size: {}'.format(voxel_size))
    pcd = pcd.voxel_down_sample(voxel_size)

    if recon_type == 'Poisson':  # poisson reconstruction
        print('run Poisson surface reconstruction')
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    elif recon_type == 'Alpha':  # Alpha
        print("Alpha reconstruction")
        alpha = 60
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    elif recon_type == 'BallPivoting':
        radii = [0.01, 0.02, 0.04, 10]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
    else:
        raise KeyError('ERROR recon_type: {}'.format(recon_type))

    if is_vis:
        show_pcd(mesh, window_name='mesh')

    # save mesh
    print('saving point cloud: {}'.format(output_mesh_path))
    o3d.io.write_triangle_mesh(output_mesh_path, mesh)

    print('MESH RECON FINISHED! ')

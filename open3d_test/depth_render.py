# Author: ANTenna 2021/12/20 6:35 下午

import os
import sys
import numpy as np
from PIL import Image
import cv2
import open3d as o3d

from tqdm import tqdm
from corerender.mesh_renderer.mesh_renderer_ply import *

from plyfile import PlyData, PlyElement
import yaml

sys.path.append("/opt/app/Utility")


# POISSON_RECON_PATH='/opt/poisson-reconstruction-11.02/Bin/Linux'


# def load_K_from_file(filename):
#     # K_: 3x3
#     with open(filename) as f:
#         f_read = f.readlines()[0]
#         K_ = np.array([np.float(item) for item in f_read.strip().split(';')])[:9].reshape((3,3))
#         image_size_ = np.array([np.float(item) for item in f_read.strip().split(';')])[9:11].reshape(2)
#         iimage_size = image_size_.astype(np.int)
#         print('check',iimage_size)
#     return K_, iimage_size


# def load_poses_from_file(pose_filename, is_mesh2pano):
#     # replace as loadtxt()
#     poses = []
#     with open(pose_filename) as f:  # each line, each view
#         for line in f:
#             ext, pose_t, pose_r, fn = line.strip().split(';')
#             fn = fn.split('/')[-1]
#             pose_t = np.array([np.float(item) for item in pose_t.split(',')])
#             pose_r = np.array([np.float(item) for item in pose_r.split(',')]).reshape((3,3))
#
#             V = np.eye(4)
#             if is_mesh2pano == True:
#                 V[:3,:3] = pose_r
#                 V[:3, -1] = -pose_r.dot(pose_t)  # convert to mesh2pano
#             else:
#                 V[:3,:3] = pose_r.transpose()
#                 V[:3, -1] = pose_t
#             poses.append((V,fn))
#     # V: (camera, target)
#     return poses


def intrinsics_to_P(intrinsics, w, h):
    fu = intrinsics[0, 0]
    fv = intrinsics[1, 1]
    u0 = intrinsics[0, 2]
    v0 = intrinsics[1, 2]
    znear, zfar = 1e-5, 1e5
    M = np.zeros((4, 4), dtype=np.float32)

    right = (w - u0) * znear / fu
    left = -u0 * znear / fu
    top = (h - v0) * znear / fv
    bottom = - v0 * znear / fv

    M[0, 0] = +2.0 * znear / (right - left)
    M[2, 0] = (right + left) / (right - left)
    M[1, 1] = +2.0 * znear / (top - bottom)
    M[3, 1] = (top + bottom) / (top - bottom)
    M[2, 2] = -(zfar + znear) / (zfar - znear)
    M[3, 2] = -2.0 * znear * zfar / (zfar - znear)
    M[2, 3] = -1.0
    return M


def render_depth(mesh_file_path, image_size, cam_K, cam_pose,
                 rgb_path, depth_output_path):
    """
    :param: mesh_file_path, camera_path
    :return: depth image aligned to rgb image
    """
    print('init render ... ')
    # renderer = SimpleMeshRenderer(width=image_size[0]//2, height=image_size[1]//2, render_marker = False)
    renderer = SimpleMeshRenderer(width=image_size[0], height=image_size[1], render_marker=False)
    print('loading mesh file: {} ... '.format(mesh_file))
    renderer.load_object(mesh_file_path)

    # camera intrinsics
    renderer.P = intrinsics_to_P(cam_K, image_size[0], image_size[1])

    # for idx in tqdm(range(len(poses))):  # for each channel
    if 1:  # test one frame
        # camera pose
        e = np.eye(4)
        e[0, 0] = -1
        e[1, 1] = -1
        e[2, 2] = -1
        # renderer.V = e.dot(poses[idx][0])  # reset camera pose
        renderer.V = e.dot(cam_pose)  # reset camera pose

        frame = renderer.render()
        # read image, poses[idx][1] contains image file name
        # img = np.array(Image.open(rgb_path))
        # save selected RGB
        # Image.fromarray(img).resize((image_size[0]//2, image_size[1]//2)).save(os.path.join(base_path, 'color_map/rgb_selected/{}_{}/{:05d}.jpg'.format(i, j, idx)))
        depth = (-1000 * frame[2][:, ::-1, 0]).astype(np.int32)  # unit: mm
        # save depth
        Image.fromarray(depth).save(depth_output_path)

        return depth


def get_cam_K_pose(f_path):
    intri_yaml_file_path = os.path.join(f_path, 'intrinsic-calib.yml')
    extrin_yaml_file_path = os.path.join(f_path, 'extrinsic-calib.yml')

    intrinsic_params = cv2.FileStorage(intri_yaml_file_path, flags=cv2.FILE_STORAGE_READ)
    # intrinsic parameter K
    intrinsic_matrix = intrinsic_params.getNode('camera_matrix').mat()
    print('intrinsic_matrix: \n {}'.format(intrinsic_matrix))
    # intrinsic parameter distortion
    distortion = intrinsic_params.getNode('distortion_coefficients').mat()
    print('distortion: \n {}'.format(distortion))
    intrinsic_params.release()

    # extrinsic parameters
    extrinsic_params = cv2.FileStorage(extrin_yaml_file_path, flags=cv2.FILE_STORAGE_READ)
    # rev
    rvec = extrinsic_params.getNode('rvec').mat()
    tvec = extrinsic_params.getNode('tvec').mat()

    rvec = np.array(list(map(lambda x: float(x), rvec)), dtype=np.float32)
    tvec = np.array(list(map(lambda x: float(x), tvec)), dtype=np.float32)

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    translation_matrix = np.array(tvec, dtype=float).reshape(3, 1)
    extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))
    print('extrinsic_matrix: \n {}'.format(extrinsic_matrix))
    # to 4x4
    cam_ext = np.eye(4)
    cam_ext[:3, :3] = rotation_matrix
    cam_ext[:3, -1] = tvec
    print('cam_ext: \n {}'.format(cam_ext))

    return intrinsic_matrix, cam_ext


if __name__ == "__main__":
    # HuanQiu rendering project for back-projection

    # input path
    root_path = '/data0/texture_data/yaohualiu/project_info/UNIVERSAL/beijing/prod/1F'
    selected_ch = 'ch05008'
    # selected_ch = 'ch05002'
    # selected_ch = 'ch04016'

    # mesh_file = 'ground_plane_mesh.ply'
    # mesh_file = '1f-crop-resampled-poisson-trimmer.ply'
    mesh_file = '1F-crop-resampled-2-poisson-trimmer.ply'

    cam_K_path = os.path.join(root_path, selected_ch, 'intrinsic-calib.yml')
    cam_pose_path = os.path.join(root_path, selected_ch, 'extrinsic-calib.yml')
    rgb_image_path = os.path.join(root_path, selected_ch, 'ref.jpg')

    # img_tmp = cv2.imread(rgb_image_path)
    # rgb_image_path = rgb_image_path.replace('.jpg', '.png')
    # cv2.imwrite(rgb_image_path, img_tmp)
    print('rgb_image_path: \n {} '.format(rgb_image_path))

    # scene_mesh_path = os.path.join(root_path, 'ground_plane_mesh.ply')  # global plane scene
    scene_mesh_path = os.path.join(root_path, mesh_file)  # global scene

    # output path for per channel
    # depth_path = os.path.join(root_path, selected_ch, 'depth.jpg')
    depth_path = os.path.join(root_path, selected_ch, 'depth_whole_render.png')
    rgbd_point_cloud_path = os.path.join(root_path, selected_ch, 'rgbd_point_cloud_frame.ply')

    # 3x3's cam K, yaml intrinsic, and 4x4 cam pose
    cam_K, cam_pose = get_cam_K_pose(f_path=os.path.join(root_path, selected_ch))

    # read image
    # rgb_image = np.array(Image.open(rgb_image_path))
    # image_size = rgb_image.shape
    image_size = [2560, 1440]
    # image_size = img_tmp.shape
    print('image_size: \n{} '.format(image_size))

    print('rendering ... ')
    # mesh_file_path, image_size, cam_K, cam_pose, rgb_image, depth_output_path
    depth_img = render_depth(mesh_file_path=scene_mesh_path,
                             image_size=image_size,
                             cam_K=cam_K,
                             cam_pose=cam_pose,
                             rgb_path=0,
                             depth_output_path=depth_path)
    # print('render depth_img: \n {}'.format(depth_img))
    print('rendered depth image shape: \n {}'.format(depth_img.shape))

    # fusion rgbd
    depth = o3d.io.read_image(depth_path)
    color = o3d.io.read_image(rgb_image_path)

    # print('depth: \n', np.asarray(depth))
    # print('color: \n', np.asarray(color))
    # print('depth_type: \n', type(depth))

    print('generating RGBD image ')
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, convert_rgb_to_intensity=False, depth_trunc=15.0)

    # rgbd to point cloud, to check result
    # rgbd to point[xyz+rgb]
    K = cam_K
    print('RGBD to point cloud ')
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(image_size[0] // 2, image_size[1] // 2,
                                          K[0, 0] / 2.0, K[1, 1] / 2.0,
                                          K[0, 2] / 2.0, K[1, 2] / 2.0))
    print('saving point cloud ... ')
    o3d.io.write_point_cloud(rgbd_point_cloud_path, pcd)

    print('RENDER FINISHED!')

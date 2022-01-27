"""
Author: ANTenna on 2021/12/20 8:13 下午
aliuyaohua@gmail.com

Description:
read opencv camera calibration file
"""

import os
import yaml
import cv2
import numpy as np
import xml.etree.ElementTree as ET


if __name__ == '__main__':
    root_path = '/Users/aibee/PycharmProjects/pythonProject/test/open3d_test/yaml_file'

    intri_yaml_file_path = os.path.join(root_path, 'intrinsic-calib.yml')
    extrin_yaml_file_path = os.path.join(root_path, 'extrinsic-calib.yml')

    intrinsic_params = cv2.FileStorage(intri_yaml_file_path, flags=cv2.FILE_STORAGE_READ)
    # K
    intrinsic_matrix = intrinsic_params.getNode('camera_matrix').mat()
    print('intrinsic_matrix: \n', intrinsic_matrix)
    # distortion
    distortion = intrinsic_params.getNode('distortion_coefficients').mat()
    print('distortion: \n', distortion)
    # intrinsic_params.release()

    # --
    # extrinsic_params_file_root = ET.parse(extrin_yaml_file_path).getroot()
    # rvec = extrinsic_params_file_root.findall('rvec')[0].text.lstrip().rstrip().split(' ')
    # tvec = extrinsic_params_file_root.findall('tvec')[0].text.lstrip().rstrip().split(' ')

    extrinsic_params = cv2.FileStorage(extrin_yaml_file_path, flags=cv2.FILE_STORAGE_READ)
    # rev
    rvec = extrinsic_params.getNode('rvec').mat()
    tvec = extrinsic_params.getNode('tvec').mat()

    rvec = np.array(list(map(lambda x: float(x), rvec)), dtype=np.float32)
    tvec = np.array(list(map(lambda x: float(x), tvec)), dtype=np.float32)

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    translation_matrix = np.array(tvec, dtype=float).reshape(3, 1)
    extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))

    print('extrinsic_matrix: \n', extrinsic_matrix)
    cam_ext = np.eye(4)
    cam_ext[:3, :3] = rotation_matrix
    cam_ext[:3, -1] = tvec
    print('cam_ext: \n', cam_ext)

    print()


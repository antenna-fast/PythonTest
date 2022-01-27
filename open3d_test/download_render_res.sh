#selected_ch='ch05008'
#selected_ch='ch05002'
#selected_ch='ch03010'
#selected_ch='ch03014'
#selected_ch='ch03016'
selected_ch='ch04002'
#selected_ch='ch04008'

remote_path='/data0/texture_data/yaohualiu/project_info/UNIVERSAL/beijing/prod/1F'
local_path="/Users/aibee/PycharmProjects/pythonProject/test/open3d_test/pcd_file/${selected_ch}"
mkdir ${local_path}


# down original pcd
# 1. plane point cloud
#scp root@192.168.90.101:${remote_path}/ground_plane.ply /Users/aibee/PycharmProjects/pythonProject/test/open3d_test/pcd_file/
# 2. cropped point cloud
#scp root@192.168.90.101:${remote_path}/1F-crop.ply /Users/aibee/PycharmProjects/pythonProject/test/open3d_test/pcd_file/

# download reconstructed mesh
#scp root@192.168.90.101:${remote_path}/1F-crop_mesh.ply /Users/aibee/PycharmProjects/pythonProject/test/open3d_test/pcd_file/

# download rgbd aligned point cloud and reconstructed mesh
# 0. reconstructed mesh
#scp root@192.168.90.101:${remote_path}/${selected_ch}/1F_mesh.ply ${local_path}/

# 1. ground plane
#scp root@192.168.90.101:${remote_path}/${selected_ch}/rgbd_point_cloud.ply ${local_path}/
# 2. real whole frame
scp root@192.168.90.101:${remote_path}/${selected_ch}/pcd_whole.ply ${local_path}/

# down yaml
#scp root@192.168.90.101:${remote_path}/${selected_ch}/'extrinsic-calib.yml' /Users/aibee/PycharmProjects/pythonProject/test/open3d_test/yaml_file/
#scp root@192.168.90.101:${remote_path}/${selected_ch}/'intrinsic-calib.yml' /Users/aibee/PycharmProjects/pythonProject/test/open3d_test/yaml_file/

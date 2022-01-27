#mesh_file_root='/Users/aibee/PycharmProjects/pythonProject/test/open3d_test/pcd_file'

# after poisson and trim
# without normals
#mesh_file_root='/Users/aibee/Downloads/1f-crop-resampled-poisson-trimmer.ply'
# add normals
mesh_file_root='/Users/aibee/Downloads/1F-crop-resampled-2-poisson-trimmer.ply'
remote_root='/data0/texture_data/yaohualiu/project_info/UNIVERSAL/beijing/prod/1F'

#scp ${mesh_file_root}/ground_plane_mesh.ply root@192.168.90.101:/data0/texture_data/yaohualiu/project_info/ANTennaDepthRender
scp ${mesh_file_root} root@192.168.90.101:${remote_root}

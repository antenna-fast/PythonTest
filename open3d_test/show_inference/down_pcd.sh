remote_root='/data0/texture_data/yaohualiu/PublicDataset/s3dis/results/pred_pcd'
#method='pointNN_graph_position_20'
method='pointNN_graph_position_20_color_norm'
mkdir "pred_pcd/${method}"
idx=0
scp -r root@192.168.90.101:"${remote_root}"/"${method}"/*"${idx}".ply  pred_pcd/${method}

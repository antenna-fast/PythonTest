
# 字符串分割
#test_str='1,2,3,4,5'
test_str='/ssd/yaohualiu/data/buc3.0_feature_old_thres_full_vt/K11_wuhan_gg_20200615_3559_7.5.0/1min_batch_did_False_face_False_top30_with_st,/ssd/yaohualiu/data/buc3.0_feature_old_thres_full_vt/K11_wuhan_gg_20210523_3559_7.5.0/1min_batch_did_False_face_False_top30_with_st,/ssd/yaohualiu/data/buc3.0_feature_old_thres_full_vt/K11_wuhan_hk_20210523_3559_7.5.0/1min_batch_did_False_face_False_top30_with_st,/ssd/yaohualiu/data/buc3.0_feature_old_thres_full_vt/K11_guangzhou_artfull_20210614_3559_7.5.0/1min_batch_did_False_face_False_top30_with_st,/ssd/yaohualiu/data/buc3.0_feature_old_thres_full_vt/K11_guangzhou_artfull_20201230_7.5.0/1min_batch_did_False_face_False_top30_with_st,/ssd/yaohualiu/data/buc3.0_feature_old_thres_full_vt/GZCI_guangzhou_mowgz_20210701_3559_7.5.0/1min_batch_did_False_face_False_top30_with_st,/ssd/yaohualiu/data/buc3.0_feature_old_thres_full_vt/ZHDC_foshan_hyc_20201219_7.5.0_v16/1min_batch_did_False_face_False_top30_with_st,/ssd/yaohualiu/data/buc3.0_feature_old_thres_full_vt/ZHDC_foshan_hyc_20200625_3559_7.5.0/1min_batch_did_False_face_False_top30_with_st'

OLD_IFS="${IFS}"
IFS=","
arr=($test_str)
IFS="$OLD_IFS"

len_val_data=${#arr[@]}
echo 'len_val_data: ' "${len_val_data}"

# 数组迭代
for((i=0;i<len_val_data;i++)) do
  echo "${i}" "${arr[i]}";
done ;

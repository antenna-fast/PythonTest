1.2
3.4
set -e
set -x  # 显示指令 及其参数

#!/bin/zsh
# 约定的标记，它告诉系统这个脚本需要什么解释器来执行，即使用哪一种 Shell

echo "Hello World !"  # echo 命令用于向窗口输出文本

# 变量名外面加不加花括号都可以 最好加上，因为可以分清边界
name_id=1
echo "name_id is "${name_id}

# 只读变量
readonly face=123
echo "face_id is "${face}

# 删除变量
# 删除后不可以再使用，不可以删除只读变量
unset face

# for语句
for skill in Ada Coffe Action Java; do
  echo "I'm good at ${skill} Script"
done

# 字符串

# 双引号
your_name='yaohua'
str="Hello, my name is \"${your_name}\" "
echo ${str}


# int to str
a=123.456
echo name is "${a}"

# 引号
single='I am yaohua!!!'
echo "${single}"  # 双引号可以使用变量
echo '${single}'  # 单引号只能原样输出, 不能出现一个单独的引号

# 拼接
echo 'my ''name'

# 字符串长度
echo 'len_single: '${#single}

# 提取子字符串
echo "${single:1:4}"

# 字符串分割
#test_str='1,2,3,4,5'
test_str='/ssd/yaohualiu/data/buc3.0_feature_old_thres_full_vt/K11_wuhan_gg_20200615_3559_7.5.0/1min_batch_did_False_face_False_top30_with_st,/ssd/yaohualiu/data/buc3.0_feature_old_thres_full_vt/K11_wuhan_gg_20210523_3559_7.5.0/1min_batch_did_False_face_False_top30_with_st,/ssd/yaohualiu/data/buc3.0_feature_old_thres_full_vt/K11_wuhan_hk_20210523_3559_7.5.0/1min_batch_did_False_face_False_top30_with_st,/ssd/yaohualiu/data/buc3.0_feature_old_thres_full_vt/K11_guangzhou_artfull_20210614_3559_7.5.0/1min_batch_did_False_face_False_top30_with_st,/ssd/yaohualiu/data/buc3.0_feature_old_thres_full_vt/K11_guangzhou_artfull_20201230_7.5.0/1min_batch_did_False_face_False_top30_with_st,/ssd/yaohualiu/data/buc3.0_feature_old_thres_full_vt/GZCI_guangzhou_mowgz_20210701_3559_7.5.0/1min_batch_did_False_face_False_top30_with_st,/ssd/yaohualiu/data/buc3.0_feature_old_thres_full_vt/ZHDC_foshan_hyc_20201219_7.5.0_v16/1min_batch_did_False_face_False_top30_with_st,/ssd/yaohualiu/data/buc3.0_feature_old_thres_full_vt/ZHDC_foshan_hyc_20200625_3559_7.5.0/1min_batch_did_False_face_False_top30_with_st'

OLD_IFS="${IFS}"
IFS=","
arr=($test_str)
IFS="$OLD_IFS"

len_val_data=${#arr[@]}
echo 'len_val_data: ' "${len_val_data}"

#for s in "${arr[@]}"
#do
#  echo "${s}"
#done

# 数组迭代
for((i=0;i<len_val_data;i++)) do
  echo "${i}" "${arr[i]}";
done ;

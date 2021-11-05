import numpy as np

gt_label = np.array([1, 0, 0, 1, 0])
pred = np.array([1, 0, 0, 0, 1])

label_len = len(pred)

# pos edge
pos_gt_set = set(np.where(gt_label == 1)[0])
pos_pred_set = set(np.where(pred == 1)[0])

# neg edge
neg_gt_set = set(np.where(gt_label == 0)[0])
neg_pred_set = set(np.where(pred == 0)[0])

# TP set is pos_gt & pos_pred
tp_set = pos_gt_set & pos_pred_set
tp_set_len = len(tp_set)

tp_tn = sum(pred == gt_label)
tn_set_len = tp_tn - tp_set_len

# FP is the intersect of neg_gt & pos_pred
fp_set = neg_gt_set & pos_pred_set
fp_set_len = len(fp_set)

# FN is neg_pred & pos_gt
fn_set = neg_pred_set & pos_gt_set
fn_set_len = len(fn_set)

# print('TP is ', tp_set)  # gt == pred == 1  TP
# print('FP set ', fp_set)

acc = tp_tn / label_len
precision = tp_set_len / (tp_set_len + fp_set_len)
recall = tp_set_len / (tp_set_len + fn_set_len)  # or /nonzero(label == 1)

print('acc is ', acc)  # 准确率
print('precision is ', precision)  # 精确率
print('recall is ', recall)  #

from sklearn import metrics
import numpy as np
import faiss

pred = np.zeros(5)
labels = np.zeros(5)


# 1. threshold selection

# roc
fpr, tpr, threshold = metrics.roc_curve(y_score=pred, y_true=labels, pos_label=1)
# TODO：绘制threshold-fpr 以及threshold-tpr曲线，在同一指标下输出不同阈值，用于计算end2end结果
# 自动化: 根据precision检索阈值，放到后续流程

# metrics.plot_roc_curve()  # need cls

# ap
average_precision = metrics.average_precision_score(y_score=pred, y_true=labels, pos_label=1)


# 2. feature metrics
# f1 score
f1_score = metrics.f1_score(y_pred=pred, y_true=labels, pos_label=1)

# 在这之前用knn进行检索
# top k acc
top_k_score = metrics.top_k_accuracy_score(y_score=pred, y_true=labels, k=5)


# knn
# input: Nxd feature
# output: Nxk相似的索引
# knn with spatial/temporal constrain

index = faiss.IndexFlatIP(reid_feat_dim)  # get reid similarity
reid_array = np.array(normed_vt_reids, np.float32)
index.add(reid_array)
sims, nbrs = index.search(reid_array, k=top_k)
knns = []



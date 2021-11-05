import dgl
import torch


# get sub_graph

def get_src(g, anchor):
    frontier = dgl.in_subgraph(g, anchor)  # anchor is dst
    src_node_idx = frontier.all_edges()[0]  # get src
    return src_node_idx


src = torch.LongTensor([1, 2, 3, 4, 5, 6])
dst = torch.LongTensor([0, 0, 0, 1, 2, 3])
g = dgl.graph((src, dst))

# 一阶邻居(src)
src_node_1 = get_src(g, [0])  # get src
print('src_node_1 is ', src_node_1)  # 统计这些index里面对应的pid，都有什么类别的

# 二阶邻居(src)
src_node_2 = get_src(g, src_node_1)  # get src
print('src_node_2 is ', src_node_2)


# node scale 统计邻居正负样本个数 精确到每一种负类别
# return: pos_link_ratio, neg_ling_ratio, neg_link_bin_count
def get_pn_ratio(g, dst_idx, vt_list, gt_vtp):
    anchor_pid = gt_vtp[vt_list[dst_idx]]
    p = 0
    n = 0

    src_node_idx = get_src(g, dst_idx)
    num_nbhs = len(src_node_idx)

    nbhd_pid_dic = {}

    if num_nbhs:
        for idx in src_node_idx:
            nbhd_pid = gt_vtp[vt_list[idx]]
            if nbhd_pid == anchor_pid:
                p += 1
            else:
                n += 1

                try:  # get bin count of neg link
                    nbhd_pid_dic[nbhd_pid] += 1
                except:
                    nbhd_pid_dic[nbhd_pid] = 1

        return p / num_nbhs, n / num_nbhs, nbhd_pid_dic


# batch scale, 也就是反应在整个图上的链接关系
# 一个基本的假设，pid类别很多（考虑这个是因为涉及到结果可视化：如何更加直观有效地展示）
def get():

    return 0

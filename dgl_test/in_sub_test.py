import dgl
import torch


g = dgl.graph((torch.tensor([0, 1, 2, 1]), torch.tensor([1, 2, 0, 0])))

frontier = dgl.in_subgraph(g, 0)  # anchor is dst
src_node_idx = frontier.all_edges()  # get src

print(src_node_idx)
print(g)

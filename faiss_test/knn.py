import faiss
import torch
import numpy as np
import time


a = np.array([[0, 0],
              [1, 1],
              [3, 3],
              [5, 5]], dtype=np.float32)
a = np.ones((1000000, 3), dtype=np.float32)
num_points = a.shape[0]
data_dim = a.shape[-1]

print("Origional data: {}".format(a.shape))
top_k = 20

s_time = time.time()
res = faiss.StandardGpuResources()  # use a single GPU
index_cpu = faiss.IndexFlatL2(data_dim)  # build a flat(CPU) L2
gpu_index = faiss.index_cpu_to_gpu(res, 0, index_cpu)  # To GPU [res, gpu_id, index(cpu)]

# index = faiss.IndexFlatIP(2)  # build the index inner product: 越大越相似, norm之后就是cos similarity
print("index is trained: ", gpu_index.is_trained)
gpu_index.add(a)  # add vector to the index
sims, nbrs = gpu_index.search(a, k=top_k)

e_time = time.time()

print('sims is \n', sims)  # L2 diatance
print('nbrs is \n', nbrs)

print("time:", e_time - s_time)

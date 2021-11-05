import faiss
import numpy as np

a = np.array([[1, 0],
              [0.9, 0],
              [0, 1.5],
              [0, 1.6]], dtype=np.float32)

print(a)

top_k = 2

# index = faiss.IndexFlatL2(2)  # build the index L2: 越小越相似
index = faiss.IndexFlatIP(2)  # build the index inner product: 越大越相似, norm之后就是cos similarity
index.add(a)  # add vector to the index
sims, nbrs = index.search(a, k=top_k)

print('sims is \n', sims)
print('nbrs is \n', nbrs)

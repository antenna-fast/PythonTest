import numpy as np

a = np.array([('yh', 100, 600),
              ('hh', 200, 700),
              ('yy', 300, 500)],
             dtype=[('name', 'U10'), ('score', '<i4'), ('perf', '<i8')])

print(a)

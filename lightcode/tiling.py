import numpy as np
import math

a = np.random.randint(1, 10)
b = np.random.randint(1, 10)
c = np.random.randint(1, 10)

m1_shape = (a,b)
m2_shape = (b,c)
m1 = np.random.rand(*m1_shape)
m2 = np.random.rand(*m2_shape)
ground_truth = m1 @ m2

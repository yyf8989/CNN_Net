import numpy as np

x = np.array(range(12)).reshape((2, 2, 3))
print(x)
# 打印array reshaoe之后的形状


y = np.reshape(x, (-1, 6))
print(y)

print()

import matplotlib.pyplot as plt

a = [1, 2, 3, 4]
b = [2, 4, 6, 8]

plt.plot(a, b)
plt.show()

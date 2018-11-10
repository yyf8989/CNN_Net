import numpy as np

x = np.array(range(12)).reshape((2,2,3))
print(x)


y = np.reshape(x, (-1, 6))
print(y)

print()
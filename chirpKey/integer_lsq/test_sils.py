import time

import matlab.engine
import numpy as np

start1 = time.time()
eng = matlab.engine.start_matlab()
end1 = time.time()
print("time", end1 - start1)

length = 10
B = np.random.normal(0, 1, (length, length))
B = matlab.double(B)
y = np.random.normal(0, 1, (length, 1))
y = matlab.double(y)
p = 1

start = time.time()
x = eng.sils(B, y, p)
end = time.time()
print("time", end - start)
print(x)

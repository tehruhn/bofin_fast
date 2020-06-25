# sudo nvidia-modprobe -u
# nvidia-smi
# cupy-cuda101
import cupy as cp
import time

iterations = 9680000
a = cp.random.rand(44,20)
b = cp.random.rand(20,1)

def ab(a,b,iterations):
    for i in range(iterations):
        cp.matmul(a,b,out=None)

t1 = time.time()
ab(a,b,iterations)
t2 = time.time()
total = t2-t1
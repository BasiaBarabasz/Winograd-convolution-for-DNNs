import numpy as np
import Winograd_convolution as wc
import sys
import Direct_convolution as Dir
import sympy as sp

#points = (0,-1/2,-1,1,2)

def random_values(size, precision):
    output = np.random.normal(-1,1,(size,size)).astype(precision)
    return output

p = sys.argv[1]
points = tuple(p.split(','))
points = tuple(sp.S(i) for i in points)
n_kernel = int(sys.argv[2])
n_output = int(sys.argv[3])
n_input = n_kernel + n_output - 1
precision = sys.argv[4]
if precision == "fp32":
    precision = np.float32
if precision == "fp16":
    precision = np.float16
if precision == "fp64":
    precision = np.float64

error_dir_all = 0
error_win_all = 0

Loop = 2000
for i in range(Loop):
    H = random_values(n_kernel, precision)
    X = random_values(n_input, precision)

    win_conv = wc.Winograd_convolution(points, H, X, precision)
    dir_conv = Dir.direct_convolution(H,X,precision)
    dir_conv_fp64 = Dir.direct_convolution(H,X,np.float64)

    error_eucl_dir = np.sqrt(np.sum(np.power((dir_conv - dir_conv_fp64),2)))
    error_eucl_win = np.sqrt(np.sum(np.power((win_conv - dir_conv_fp64),2)))

    error_dir_all = error_dir_all + error_eucl_dir
    error_win_all = error_win_all + error_eucl_win

error_dir_all = error_dir_all/(Loop*n_output*n_output)
error_win_all = error_win_all/(Loop*n_output*n_output)

print("Average Euclidean error per single output point over "+str(Loop)+" loops")
print("Direct convolution: ", error_dir_all)
print("Winograd convolution: ", error_win_all)


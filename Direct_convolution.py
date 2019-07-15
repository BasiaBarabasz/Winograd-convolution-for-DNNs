import numpy as np

def direct_convolution(kernel, inputt, precision):
    kernel = np.array(kernel).astype(precision)
    inputt = np.array(inputt).astype(precision)
    n_kernel = kernel.shape[0]
    n_output = inputt.shape[0] - n_kernel + 1
    direct = np.zeros((n_output,n_output)).astype(precision)
    for inp1 in range(n_output):
        for inp2 in range(n_output):
            for kern1 in range(n_kernel):
                for kern2 in range(n_kernel):
                    direct[inp1,inp2] = direct[inp1,inp2] + (kernel[kern1,kern2] * inputt[inp1+kern1,inp2+kern2])
    return direct




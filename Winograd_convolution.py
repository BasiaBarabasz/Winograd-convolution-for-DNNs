import numpy as np
import Toom_Cook_Matrices as TC
import sys



def input_transform(BT, B, inputt):
    mul1 = BT.dot(inputt)
    result = mul1.dot(B) 
    return result   

def kernel_transform(G, GT, kernel):
    mul1 = G.dot(kernel)
    result = mul1.dot(GT)
    return result

def Hadamard_product(transformed_input, transformed_kernel):
    result = np.multiply(transformed_input, transformed_kernel)
    return result

def output_transform(AT, A, Hadamard_product):
    mul1 = AT.dot(Hadamard_product)
    result = mul1.dot(A)
    return result

def Winograd_convolution(points, H, X, precision):
    n_kernel = H.shape[0]
    n_input = X.shape[0]
    n_output = n_input - n_kernel + 1
    G, GT, AT, A, BT, B = TC.Toom_Cook_Matrices(points, n_kernel, n_output, precision)
    kernel_transformed = kernel_transform(G, GT, H)
    input_transformed = input_transform(BT, B, X)
    Hadamard = Hadamard_product(kernel_transformed, input_transformed)
    result = output_transform(AT, A, Hadamard)
    return result








from functools import reduce
import operator
import sympy as sp
import numpy as np
from sympy.physics.quantum import TensorProduct
sp.init_printing()

zero=sp.S('0')
one=sp.S('1')

def G_Matrix(points,n_p,n_kern):
    N=tuple(sp.S('1')/reduce(operator.mul, (points[i]-points[j] for j in range(n_p) if i != j)) for i in range(n_p))
    G=sp.Matrix([[points[i]**j*N[i] for j in range(n_kern)] for i in range(n_p)])
    row=sp.Matrix([[zero for j in range(n_kern-1)]+[one]])
    G=G.row_insert(n_p,row)
    G=G.expand()
    GT=sp.transpose(G)
    return G,GT

def AT_Matrix(points,n_p, n_out):
    AT=sp.Matrix([[points[j]**i for j in range(n_p)] for i in range(n_out)])
    col=sp.Matrix([zero for i in range(n_out-1)]+[one])
    AT=AT.col_insert(n_p,col)
    A=sp.transpose(AT)
    return AT,A

def BT_Matrix(points,n_p):
    m=tuple(sp.S('x')-points[i] for i in range(n_p))
    M=reduce(operator.mul, m).expand()
    Mi=tuple(reduce(operator.mul, (m[j] for j in range(n_p) if i!=j)).expand() for i in range(n_p))
    BT=sp.Matrix([[mi.coeff(sp.S('x'),j) for j in range(n_p)]+[zero] for mi in Mi])
    row=sp.Matrix([[M.coeff(sp.S('x'),j) for j in range(n_p+1)]])
    BT=BT.row_insert(n_p,row)
    B=sp.transpose(BT)

    return BT,B

def Toom_Cook_Matrices(points, kernel_size, output_size, precision):
    input_size = kernel_size + output_size - 1
    points_number = input_size - 1
    G,GT = G_Matrix(points, points_number, kernel_size)
    AT,A = AT_Matrix(points, points_number, output_size)
    BT,B = BT_Matrix(points, points_number)
    G = np.array(G).astype(precision)
    GT = np.array(GT).astype(precision)
    AT = np.array(AT).astype(precision)
    A = np.array(A).astype(precision)
    BT = np.array(BT).astype(precision)
    B = np.array(B).astype(precision)
    return G, GT, AT, A, BT, B

def Toom_Cook_Kronecker_Matrices(points, kernel_size, output_size,precision):
    input_size = kernel_size + output_size - 1
    points_number = input_size - 1
    G, GT = G_Matrix(points, points_number, kernel_size)
    AT, A = AT_Matrix(points, points_number, output_size)
    BT, B = BT_Matrix(points, points_number)
    G_Kronecker = TensorProduct(G,G)
    AT_Kronecker = TensorProduct(AT,AT)
    BT_Kronecker = TensorProduct(BT,BT)
    G_Kronecker = np.array(G_Kronecker).astype(precision)
    AT_Kronecker = np.array(AT_Kronecker).astype(precision)
    BT_Kronecker = np.array(BT_Kronecker).astype(precision)
    print(G_Kronecker)
    return G_Kronecker, AT_Kronecker, BT_Kronecker


#Toom_Cook_Kronecker_Matrices((0,-1/2,-1,1,2),3,4, np.float32)

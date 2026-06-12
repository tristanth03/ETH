import numpy as np
import scipy.sparse as sp


def assembleMatrix(N, a, b, A, B, C):
    # Inputs:
    # - N: number of interior grid points (= dim of the space VN)
    # - a,b: such that G = (a,b)
    # - A, B and C: functions from G to IR.

    # Output:
    # Approximation of NxN Matrix M defined by
    # M_{ij} := int_a^b b_i'(s) b_j'(s) A(s) + b_i(s)b_j'(s) B(s) + b_i(s)b_j(s)C(s) ds

    # Assumes a uniform mesh x_i = a + (b-a)/(N+1)*i, 0 <= i <= N+1
    # and Dirichlet conditions at a and b (i.e. b_i are the pw linear functions
    # defined by b_i(x_j) = delta_{ij}, 1 <= i,j <= N)

    MA = assembleMatrixA(N, a, b, A)
    MB = assembleMatrixB(N, a, b, B)
    MC = assembleMatrixC(N, a, b, C)

    return MA + MB + MC

# Uniform mesh of G= (a,b)


def xgrid(N, a, b):
    return np.linspace(a, b, N+2)

# h = (b-a)/(N+1)


def hval(N, a, b):
    return (b-a)/(N+1)


def assembleMatrixA(N, a, b, fun):
    # Inputs:
    # - N: number of interior grid points (= dim of the space VN)
    # - R: such that G = (0,R)
    # - fun: G -> IR, s -> f(s), function defined on G

    # Output:
    # Approximation of NxN Matrix A defined by
    # A_{ij} := int_0^R b_i'(s) b_j'(s) f(s) ds

    xi = xgrid(N, a, b)  # xi = [x0,...,xN+1]
    h = hval(N, a, b)

    diag = np.zeros(N)

    p1 = h/2 - h/(2*np.sqrt(3))  # Gauss points
    p2 = h/2 + h/(2*np.sqrt(3))
    w1 = h/2  # Gauss weights
    w2 = h/2

    dN1 = 1/h  # Shape function derivative
    dN2 = -1/h  # Shape function derivative

    diag = w1*dN1*dN1*fun(xi[:-2] + p1) + w2*dN1*dN1*fun(xi[:-2] + p2)
    diag += w1*dN2*dN2*fun(xi[1:-1] + p1) + w2*dN2*dN2*fun(xi[1:-1]+p2)

    lodiag = w1*dN1*dN2*fun(xi[1:-2]+p1) + w2*dN1*dN2*fun(xi[1:-2]+p2)
    updiag = lodiag

    A = sp.diags([lodiag, diag, updiag], [-1, 0, 1])
    return A


def assembleMatrixB(N, a, b, fun):
    # Inputs:
    # - N: number of interior grid points (= dim of the space VN)
    # - R: such that G = (0,R)
    # - fun: G -> IR, s -> f(s), function defined on G

    # Output:
    # Approximation of NxN Matrix B defined by
    # B_{ij} := int_0^R b_i(s) b_j'(s) f(s) ds
    xi = xgrid(N, a, b)
    h = hval(N, a, b)

    p1 = h/2 - h/(2*np.sqrt(3))  # Gauss points
    p2 = h/2 + h/(2*np.sqrt(3))
    w1 = h/2  # Gauss weights
    w2 = h/2

    N11 = p1/h  # First shape function value at p1
    N12 = p2/h  # First shape function value at p2
    N21 = 1 - p1/h  # Second shape function value at p1
    N22 = 1 - p2/h  # Second shape function value at p2

    dN1 = 1/h  # Shape function derivative
    dN2 = -1/h  # Shape function derivative

    diag = w1*N11*dN1*fun(xi[:-2] + p1) + w2*N12*dN1*fun(xi[:-2] + p2)
    diag += w1*N21*dN2*fun(xi[1:-1] + p1) + w2*N22*dN2*fun(xi[1:-1]+p2)

    lodiag = w1*dN2*N11*fun(xi[1:-2]+p1) + w2*dN2*N12*fun(xi[1:-2]+p2)
    updiag = w1*dN1*N21*fun(xi[1:-2]+p1) + w2*dN1*N22*fun(xi[1:-2]+p2)

    B = sp.diags([lodiag, diag, updiag], [-1, 0, 1])
    return B


def assembleMatrixC(N, a, b, fun):
    # Inputs:
    # - N: number of interior grid points (= dim of the space VN)
    # - R: such that G = (0,R)
    # - fun: G -> IR, s -> f(s), function defined on G

    # Output:
    # Approximation of NxN Matrix C defined by
    # C_{ij} := int_0^R b_i(s) b_j(s) f(s) ds

    xi = xgrid(N, a, b)
    h = hval(N, a, b)

    p1 = h/2 - h/(2*np.sqrt(3))  # Gauss points
    p2 = h/2 + h/(2*np.sqrt(3))
    w1 = h/2  # Gauss weights
    w2 = h/2

    N11 = p1/h  # First shape function value at p1
    N12 = p2/h  # First shape function value at p2
    N21 = 1 - p1/h  # Second shape function value at p1
    N22 = 1 - p2/h  # Second shape function value at p2

    dN1 = 1/h  # Shape function derivative
    dN2 = -1/h  # Shape function derivative

    diag = w1*N11*N11*fun(xi[:-2] + p1) + w2*N12*N12*fun(xi[:-2] + p2)
    diag += w1*N21*N21*fun(xi[1:-1] + p1) + w2*N22*N22*fun(xi[1:-1]+p2)

    lodiag = w1*N21*N11*fun(xi[1:-2]+p1) + w2*N22*N12*fun(xi[1:-2]+p2)
    updiag = lodiag

    C = sp.diags([lodiag, diag, updiag], [-1, 0, 1])
    return C
